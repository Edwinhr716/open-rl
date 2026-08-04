"""Gate B variant: replay the Phase 3 demo's exact trainer op order with REAL
forward_backward gradients, chasing tenant-B's snapshot-mode-only NaN.

Differences vs test_trainer_swap.py (all matching the demo, where B broke):
- real cross_entropy forward_backward (driver's 4x identical datums, LR 3e-3,
  grad_clip 1.0) instead of fake_grads;
- tenant-B's AdamW state materializes on its first optim_step, which runs
  AFTER swap_out(A) has released ~2GB of GPU-CR blocks (the demo's allocation
  window; Gate B materialized both optimizers before any swap);
- every swap_in is verified bitwise against CPU copies taken at that
  tenant's swap_out, and both tenants' full tensor sets are scanned for
  non-finite values whenever they are resident;
- 2MB-block bookkeeping: if a tensor of one tenant lands in a 2MB block that
  the other tenant's snapshot covers, restores will clobber it — logged as
  CROSS-TENANT BLOCK OVERLAP.

Exit 0 = every round trained finite and every restore was bitwise-clean.
"""

import json
import os
import sys

import torch

from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.timeslice_tenant import GPU_CR_MAX_TRACKED_ADDR, TimesliceTenantManager
from training.trainer_worker import Datum

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
ROUNDS = int(os.getenv("ROUNDS", "4"))
LORA_RANK = int(os.getenv("LORA_RANK", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-3"))
PROMPT_TOKENS = list(range(1, 17))
TARGETS = {"tenant-A": [420] * 16, "tenant-B": [777] * 16}

failures: list[str] = []
GATE_LOG = os.getenv("GATE_LOG", "/tmp/gate-b-real.log")


def log(msg: str):
  print(msg, flush=True)
  os.makedirs(os.path.dirname(GATE_LOG), exist_ok=True)
  with open(GATE_LOG, "a") as f:
    f.write(msg + "\n")


def check(cond: bool, msg: str):
  log(f"[gate-b-real] {'OK ' if cond else 'FAILURE'}: {msg}")
  if not cond:
    failures.append(msg)


def _bytes(t: torch.Tensor) -> torch.Tensor:
  return t.contiguous().view(torch.uint8).flatten()


def make_datum(tenant: str) -> Datum:
  return Datum(
    model_input=PROMPT_TOKENS,
    loss_fn_inputs={"target_tokens": {"data": TARGETS[tenant]}, "weights": {"data": [1.0] * len(TARGETS[tenant])}},
  )


def tenant_tensors(worker, tenant: str) -> list[tuple[str, torch.Tensor]]:
  state = worker.adapter_states[tenant]
  opt = state["optimizer"]
  out = []
  for i, p in enumerate(state["trainable_params"]):
    out.append((f"param:{i}", p.data))
    if opt is None:
      continue
    for key in ("exp_avg", "exp_avg_sq"):
      t = opt.state.get(p, {}).get(key)
      if t is not None and t.is_cuda:
        out.append((f"{key}:{i}", t))
  return out


def block_set(tensors: list[tuple[str, torch.Tensor]]) -> dict[int, list[str]]:
  """2MB-aligned block -> tensor names touching it (GPU-CR's dump/restore granularity)."""
  blocks: dict[int, list[str]] = {}
  for name, t in tensors:
    start, end = t.data_ptr(), t.data_ptr() + t.element_size() * t.nelement()
    for b in range(start >> 21, ((end - 1) >> 21) + 1):
      blocks.setdefault(b, []).append(name)
  return blocks


def describe_addrs(tensors: list[tuple[str, torch.Tensor]], label: str):
  ptrs = [t.data_ptr() for _, t in tensors]
  untracked = sum(1 for p in ptrs if p >= GPU_CR_MAX_TRACKED_ADDR)
  log(
    f"[gate-b-real] {label}: {len(ptrs)} tensors, addr range {hex(min(ptrs))}-{hex(max(ptrs))}, "
    f"{len(block_set(tensors))} distinct 2MB blocks, {untracked} above GPU_CR_MAX_TRACKED_ADDR"
  )


def nan_scan(worker, tenant: str, when: str):
  bad = [name for name, t in tenant_tensors(worker, tenant) if not torch.isfinite(t).all()]
  check(not bad, f"{tenant} tensors all finite {when}" + (f" (non-finite: {bad[:6]}{'...' if len(bad) > 6 else ''})" if bad else ""))
  return bad


def cross_overlap(parked_blocks: dict[int, list[str]], resident: list[tuple[str, torch.Tensor]], parked: str, res_tenant: str):
  """Blocks covered by the parked tenant's snapshot that also hold resident-tenant bytes."""
  hits = []
  for name, t in resident:
    start, end = t.data_ptr(), t.data_ptr() + t.element_size() * t.nelement()
    for b in range(start >> 21, ((end - 1) >> 21) + 1):
      if b in parked_blocks:
        hits.append((name, hex(b << 21), parked_blocks[b][:3]))
  if hits:
    log(f"[gate-b-real] CROSS-TENANT BLOCK OVERLAP: {len(hits)} tensors of {res_tenant} share 2MB blocks with {parked}'s snapshot:")
    for name, blk, parked_names in hits[:10]:
      log(f"    {res_tenant}/{name} in block {blk} also holding {parked}/{parked_names}")
  else:
    log(f"[gate-b-real] no 2MB-block overlap between resident {res_tenant} and parked {parked}'s snapshot")
  return hits


class Boundary:
  """Mirrors LoraTrainingRequestsProcessor's tenant-boundary handling, with forensics between the two swap legs."""

  def __init__(self, worker, manager):
    self.worker = worker
    self.manager = manager
    self.active: str | None = None
    self.refs: dict[str, dict[str, torch.Tensor]] = {}  # tenant -> CPU copies at swap_out
    self.parked_blocks: dict[str, dict[int, list[str]]] = {}

  def switch(self, next_id: str):
    prev = self.active
    self.active = next_id
    if prev == next_id:
      return
    if prev is not None and prev not in self.manager._swapped_out:
      if self.manager.addresses_for(prev):
        tensors = tenant_tensors(self.worker, prev)
        self.refs[prev] = {name: t.detach().cpu().clone() for name, t in tensors}
        self.parked_blocks[prev] = block_set(tensors)
        describe_addrs(tensors, f"{prev} swap set at swap_out")
      self.manager.swap_out(prev)
    if next_id in self.manager._swapped_out:
      self.manager.swap_in(next_id)
      torch.cuda.synchronize()
      self.verify_restore(next_id)
    if prev is not None and prev in self.parked_blocks and next_id in self.worker.adapter_states:
      cross_overlap(self.parked_blocks[prev], tenant_tensors(self.worker, next_id), prev, next_id)

  def verify_restore(self, tenant: str):
    ref = self.refs.get(tenant)
    if ref is None:
      return
    mismatches = []
    for name, t in tenant_tensors(self.worker, tenant):
      if name not in ref:
        continue
      got_b = _bytes(t.detach().cpu())
      ref_b = _bytes(ref[name])
      if torch.equal(got_b, ref_b):
        continue
      diff = (got_b != ref_b).nonzero().flatten()
      got = t.detach().cpu()
      mismatches.append((name, hex(t.data_ptr()), int((~torch.isfinite(got)).sum()), got.flatten()[:4].tolist(),
                         int(diff[0]), int(diff[-1]), int(diff.numel()), got_b))
    check(not mismatches, f"{tenant} restore byte-identical ({len(ref)} tensors, {len(mismatches)} mismatches)")
    for name, ptr, n_nonfinite, sample, first, last, nbytes, got_b in mismatches[:16]:
      src = self.find_source(got_b, exclude=f"{tenant}/{name}")
      log(f"    MISMATCH {tenant}/{name} @ {ptr}: bytes[{first}..{last}] differ ({nbytes} bytes), "
          f"{n_nonfinite} non-finite, head={sample}, restored-bytes-equal={src}")

  def find_source(self, got_b: torch.Tensor, exclude: str) -> str:
    """Does this restored payload exactly equal some OTHER tensor's snapshot bytes?"""
    for tenant, refs in self.refs.items():
      for name, r in refs.items():
        if f"{tenant}/{name}" == exclude:
          continue
        rb = _bytes(r)
        if rb.numel() == got_b.numel() and torch.equal(rb, got_b):
          return f"{tenant}/{name}"
    return "no-other-block"


def main() -> int:
  log(f"[gate-b-real] {BASE_MODEL} rank={LORA_RANK} lr={LEARNING_RATE} rounds={ROUNDS}")
  worker = LoraTrainingWorker()
  manager = TimesliceTenantManager(enabled=True)
  manager.attach_worker(worker)
  boundary = Boundary(worker, manager)
  cfg = LoraConfig(rank=LORA_RANK, seed=0)

  # Driver creates both models up front: boundary None->A, then A->B.
  boundary.switch("tenant-A")
  worker.create_model(BASE_MODEL, "tenant-A", cfg)
  manager.forget("tenant-A")
  boundary.switch("tenant-B")
  worker.create_adapter("tenant-B", cfg)
  manager.forget("tenant-B")

  # WARMUP_ROUNDS>0: train each tenant with NO swaps first, so every
  # persistent allocation (AdamW state, autograd/cublas workspaces) exists
  # before the first snapshot — tests the layout-churn root-cause theory.
  for w in range(int(os.getenv("WARMUP_ROUNDS", "0"))):
    for tenant in ("tenant-A", "tenant-B"):
      fb = worker.forward_backward([make_datum(tenant)] * 4, "cross_entropy", None, tenant)
      opt = worker.optim_step({"learning_rate": LEARNING_RATE, "grad_clip_norm": 1.0}, tenant)
      log(f"[gate-b-real] warmup {w} {tenant}: loss={fb['metrics']['loss:mean']} grad_norm={opt['metrics']['grad_norm:mean']}")
  boundary.active = "tenant-B"

  for r in range(ROUNDS):
    for tenant in ("tenant-A", "tenant-B"):
      boundary.switch(tenant)
      if tenant in worker.adapter_states and worker.adapter_states[tenant]["optimizer"] is not None:
        nan_scan(worker, tenant, f"before round {r} fb")
      fb = worker.forward_backward([make_datum(tenant)] * 4, "cross_entropy", None, tenant)
      loss = fb["metrics"]["loss:mean"]
      opt = worker.optim_step({"learning_rate": LEARNING_RATE, "grad_clip_norm": 1.0}, tenant)
      grad_norm = opt["metrics"]["grad_norm:mean"]
      torch.cuda.synchronize()
      finite = loss is not None and grad_norm is not None and torch.isfinite(torch.tensor([loss, grad_norm])).all()
      check(bool(finite), f"round {r} {tenant}: loss={loss} grad_norm={grad_norm}")
      nan_scan(worker, tenant, f"after round {r} optim_step")
      if r == 0:
        describe_addrs(tenant_tensors(worker, tenant), f"{tenant} full set after first optim_step")
      other = "tenant-B" if tenant == "tenant-A" else "tenant-A"
      if other in manager._swapped_out and other in boundary.parked_blocks:
        # Did this tenant's fresh allocations land inside blocks the parked
        # tenant's snapshot covers? If so, the parked tenant's restore will
        # clobber them.
        cross_overlap(boundary.parked_blocks[other], tenant_tensors(worker, tenant), other, tenant)
      worker.save_adapter(tenant, alias=f"sampler-{r}")

  summary = {"event": "gate_b_real_summary", "model": BASE_MODEL, "rank": LORA_RANK, "lr": LEARNING_RATE, "rounds": ROUNDS, "failures": len(failures)}
  manager.metrics.emit(**summary)
  log(f"[gate-b-real] SUMMARY {json.dumps(summary)}")
  log(f"[gate-b-real] {'PASSED' if not failures else 'FAILED: ' + '; '.join(failures[:8])}")
  return 0 if not failures else 1


if __name__ == "__main__":
  sys.exit(main())
