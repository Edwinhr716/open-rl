"""Phase 2 Gate B: trainer-side tenant swap correctness + VRAM reclaim.

Runs against the real LoraTrainingWorker and TimesliceTenantManager:

1. Create two tenant adapters on BASE_MODEL, give each one optim_step
   (synthetic gradients — AdamW state materializes on GPU).
2. Snapshot CPU copies of tenant-A's params + exp_avg/exp_avg_sq.
3. swap_out(A): physical VRAM must be released (mem_get_info delta > 0).
4. swap_in(A): every tensor must be bitwise-identical to the CPU copies.
5. THE critical test: another optim_step on A after restore — GPU-CR's known
   weak spot is kernels *writing* to restored memory. Params must change and
   no CUDA error may surface (CUDA_LAUNCH_BLOCKING=1 makes errors synchronous).
6. A few alternating A/B rounds for timing metrics.

Exit 0 = PASS, 1 = FAIL.
"""

import json
import os
import sys
import time

import torch

from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.timeslice_tenant import TimesliceTenantManager

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
ROUNDS = int(os.getenv("TIMESLICE_SELFTEST_ROUNDS", "3"))
LORA_RANK = int(os.getenv("LORA_RANK", "16"))

failures: list[str] = []


def check(cond: bool, msg: str):
  print(f"[gate-b] {'OK ' if cond else 'FAILURE'}: {msg}")
  if not cond:
    failures.append(msg)


def fake_grads(worker, tenant):
  torch.manual_seed(int(time.time() * 1000) % 100000)
  for p in worker.adapter_states[tenant]["trainable_params"]:
    p.grad = torch.randn_like(p) * 1e-3


def tenant_tensors(worker, tenant):
  state = worker.adapter_states[tenant]
  opt = state["optimizer"]
  out = []
  for p in state["trainable_params"]:
    out.append(("param", p.data))
    for key in ("exp_avg", "exp_avg_sq"):
      t = opt.state.get(p, {}).get(key)
      if t is not None and t.is_cuda:
        out.append((key, t))
  return out


def main() -> int:
  print(f"[gate-b] loading {BASE_MODEL} + 2 tenants (rank {LORA_RANK})")
  worker = LoraTrainingWorker()
  cfg = LoraConfig(rank=LORA_RANK, seed=0)
  worker.create_model(BASE_MODEL, "tenant-A", cfg)
  worker.create_adapter("tenant-B", cfg)

  manager = TimesliceTenantManager(enabled=True)
  manager.attach_worker(worker)

  # Materialize AdamW state for both tenants.
  for tenant in ("tenant-A", "tenant-B"):
    fake_grads(worker, tenant)
    worker.optim_step({"learning_rate": 1e-4}, tenant)
  torch.cuda.synchronize()

  n_regions = len(manager.addresses_for("tenant-A") or [])
  print(f"[gate-b] tenant-A swap set: {n_regions} regions")

  # --- 2. CPU reference copies of A ---
  ref = {f"{k}:{i}": t.detach().cpu().clone() for i, (k, t) in enumerate(tenant_tensors(worker, "tenant-A"))}

  # --- 3. swap_out(A): VRAM must actually free ---
  free_before = torch.cuda.mem_get_info()[0]
  t0 = time.monotonic()
  manager.swap_out("tenant-A")
  out_ms = (time.monotonic() - t0) * 1000
  torch.cuda.synchronize()
  freed_mb = (torch.cuda.mem_get_info()[0] - free_before) / 1e6
  check("tenant-A" in manager._swapped_out, "swap_out completed")
  check(freed_mb > 10, f"physical VRAM freed: {freed_mb:.0f}MB in {out_ms:.0f}ms")

  # --- 4. swap_in(A): bitwise equality ---
  t0 = time.monotonic()
  manager.swap_in("tenant-A")
  in_ms = (time.monotonic() - t0) * 1000
  torch.cuda.synchronize()
  mismatches = sum(1 for i, (k, t) in enumerate(tenant_tensors(worker, "tenant-A")) if not torch.equal(t.detach().cpu(), ref[f"{k}:{i}"]))
  check(mismatches == 0, f"restored tensors bitwise-identical ({len(ref)} tensors, {mismatches} mismatches) in {in_ms:.0f}ms")

  # --- 5. optim_step AFTER restore (write path on restored memory) ---
  before = worker.adapter_states["tenant-A"]["trainable_params"][0].detach().cpu().clone()
  try:
    fake_grads(worker, "tenant-A")
    worker.optim_step({"learning_rate": 1e-3}, "tenant-A")
    torch.cuda.synchronize()
    after = worker.adapter_states["tenant-A"]["trainable_params"][0].detach().cpu()
    check(not torch.equal(before, after), "optim_step after restore updated params (write-after-restore works)")
  except Exception as e:
    check(False, f"optim_step after restore raised: {e}")

  # --- 6. alternating rounds for timing ---
  switch_ms = []
  active, inactive = "tenant-A", "tenant-B"
  for r in range(ROUNDS):
    t0 = time.monotonic()
    manager.switch_tenant(active, inactive)
    ms = (time.monotonic() - t0) * 1000
    active, inactive = inactive, active
    fake_grads(worker, active)
    worker.optim_step({"learning_rate": 1e-4}, active)
    switch_ms.append(ms)
    print(f"[gate-b] round {r}: switched to {active} in {ms:.0f}ms, optim_step OK")
  torch.cuda.synchronize()

  summary = {
    "event": "gate_b_summary",
    "model": BASE_MODEL,
    "rank": LORA_RANK,
    "regions": n_regions,
    "vram_freed_mb": round(freed_mb, 1),
    "swap_out_ms": round(out_ms, 1),
    "swap_in_ms": round(in_ms, 1),
    "switch_ms": [round(m, 1) for m in switch_ms],
    "failures": len(failures),
  }
  manager.metrics.emit(**summary)
  print(f"[gate-b] SUMMARY {json.dumps(summary)}")
  print(f"[gate-b] {'PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
  return 0 if not failures else 1


if __name__ == "__main__":
  sys.exit(main())
