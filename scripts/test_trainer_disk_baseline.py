"""Multiplexed-baseline measurement: tenant switching via Open-RL's existing
disk eviction path, for comparison against snapshot-agent swaps.

When VRAM cannot hold every tenant, stock Open-RL's only eviction mechanism
is save_state(include_optimizer=True) -> drop the adapter -> later
load_from_state(restore_optimizer=True). This job times those round-trips on
the same hardware and model as Gate B / the demo, but under the baseline's
fair conditions: no LD_PRELOAD, no CUDA_LAUNCH_BLOCKING, default CUDA caching
allocator.

Per cycle and tenant: save_state + evict (delete_adapter + empty_cache,
measuring VRAM actually returned to the driver) then load_from_state,
followed by an optim_step to prove training continues. Emits the same
metrics.jsonl schema as the snapshot path (events disk_swap_out /
disk_swap_in / disk_switch).

Exit 0 = all cycles completed and post-reload optim_step worked.
"""

import json
import os
import shutil
import subprocess
import sys
import time

import torch

from server.timeslice_lora import MetricsWriter
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
CYCLES = int(os.getenv("CYCLES", "4"))
STATE_ROOT = os.getenv("STATE_ROOT", "/mnt/open-rl/disk-baseline")
LORA_RANK = int(os.getenv("LORA_RANK", "16"))
TENANTS = int(os.getenv("TENANTS", "2"))
# FLUSH_CACHES=1 removes page-cache flattery: sync after save (real write
# cost) and drop caches before load (real read cost). Needs a privileged pod.
FLUSH_CACHES = os.getenv("FLUSH_CACHES", "0") == "1"


def _flush_page_cache():
  try:
    with open("/proc/sys/vm/drop_caches", "w") as f:
      f.write("3")
    return True
  except OSError:
    return False

metrics = MetricsWriter()
failures: list[str] = []


def check(cond: bool, msg: str):
  print(f"[disk-baseline] {'OK ' if cond else 'FAILURE'}: {msg}")
  if not cond:
    failures.append(msg)


def fake_grads(worker, tenant):
  for p in worker.adapter_states[tenant]["trainable_params"]:
    p.grad = torch.randn_like(p) * 1e-3


def state_dir_mb(path: str) -> float:
  out = subprocess.run(["du", "-sk", path], capture_output=True, text=True).stdout
  return int(out.split()[0]) / 1024 if out else 0.0


def swap_out_disk(worker, tenant: str) -> dict:
  """save_state + evict from GPU; returns timings + VRAM actually freed."""
  path = os.path.join(STATE_ROOT, tenant)
  shutil.rmtree(path, ignore_errors=True)

  t0 = time.monotonic()
  worker.save_state(tenant, path, include_optimizer=True)
  if FLUSH_CACHES:
    os.sync()  # charge the real write to the save, not the page cache
  save_ms = (time.monotonic() - t0) * 1000

  free_before = torch.cuda.mem_get_info()[0]
  t0 = time.monotonic()
  other = next((t for t in worker.adapter_states if t != tenant), None)
  if other is not None:
    worker.peft_model.set_adapter(other)
  worker.peft_model.delete_adapter(tenant)
  worker.adapter_states.pop(tenant, None)
  torch.cuda.synchronize()
  torch.cuda.empty_cache()  # caching allocator: nothing returns to the driver without this
  evict_ms = (time.monotonic() - t0) * 1000
  freed_mb = (torch.cuda.mem_get_info()[0] - free_before) / 1e6

  row = {"save_ms": round(save_ms, 1), "evict_ms": round(evict_ms, 1),
         "vram_freed_mb": round(freed_mb, 1), "state_mb": round(state_dir_mb(path), 1),
         "wall_ms": round(save_ms + evict_ms, 1)}
  metrics.emit(event="disk_swap_out", lora_id=tenant, mode="disk_baseline", **row)
  return row


def swap_in_disk(worker, tenant: str) -> dict:
  path = os.path.join(STATE_ROOT, tenant)
  if FLUSH_CACHES and not _flush_page_cache():
    print("[disk-baseline] WARNING: could not drop caches; load will be cache-warm")
  t0 = time.monotonic()
  worker.load_from_state(tenant, path, restore_optimizer=True)
  torch.cuda.synchronize()
  row = {"wall_ms": round((time.monotonic() - t0) * 1000, 1)}
  metrics.emit(event="disk_swap_in", lora_id=tenant, mode="disk_baseline", **row)
  return row


def main() -> int:
  print(f"[disk-baseline] loading {BASE_MODEL} + 2 tenants (rank {LORA_RANK}), state root {STATE_ROOT}, flush={FLUSH_CACHES}")
  os.makedirs(STATE_ROOT, exist_ok=True)
  worker = LoraTrainingWorker()
  cfg = LoraConfig(rank=LORA_RANK, seed=0)
  names = [f"tenant-{i:03d}" for i in range(TENANTS)]
  # Disk-multiplex capacity mode: create each tenant, give it optimizer
  # state, then park the previous one so at most one stays resident.
  worker.create_model(BASE_MODEL, names[0], cfg)
  for i, tenant in enumerate(names):
    if i > 0:
      worker.create_adapter(tenant, cfg)
    fake_grads(worker, tenant)
    worker.optim_step({"learning_rate": 1e-4}, tenant)
    if i > 0:
      outs_boot = swap_out_disk(worker, names[i - 1])
      free_mb = torch.cuda.mem_get_info()[0] / 1e6
      metrics.emit(event="disk_tenant_added", tenant_count=i + 1, vram_free_mb=round(free_mb, 1))
      print(f"[disk-baseline] tenant {i + 1}/{TENANTS} created, parked {names[i - 1]} ({outs_boot['wall_ms']:.0f}ms), vram_free={free_mb:.0f}MB")
  torch.cuda.synchronize()

  ref = worker.adapter_states[names[-1]]["trainable_params"][0].detach().cpu().clone()

  outs, ins = [], []
  parked = None  # names[-1] is resident; everyone else is on disk
  active = names[-1]
  for c in range(CYCLES):
    for tenant in names:
      t0 = time.monotonic()
      if tenant != active:
        # restore first, then park: keeps a second adapter present for
        # set_adapter during eviction (sole-resident case) at the cost of
        # both tenants transiently resident (~3GB, fits headroom)
        ins.append(swap_in_disk(worker, tenant)["wall_ms"])
        outs.append(swap_out_disk(worker, active)["wall_ms"])
        active = tenant
      metrics.emit(event="disk_switch", lora_id=tenant, mode="disk_baseline",
                   wall_ms=round((time.monotonic() - t0) * 1000, 1))
      fake_grads(worker, tenant)
      worker.optim_step({"learning_rate": 1e-4}, tenant)
      print(f"[disk-baseline] cycle {c} {tenant}: active, optim_step OK")

  # Correctness: reload the longest-parked tenant, then round-trip it again.
  if names[0] not in worker.adapter_states:
    ins.append(swap_in_disk(worker, names[0])["wall_ms"])
  if names[0] in worker.adapter_states:
    saved = worker.adapter_states[names[0]]["trainable_params"][0].detach().cpu().clone()
    swap_out_disk(worker, names[0])
    swap_in_disk(worker, names[0])
    reloaded = worker.adapter_states[names[0]]["trainable_params"][0].detach().cpu()
    check(torch.equal(saved, reloaded), "reloaded params identical to what was saved")
    check(not torch.equal(ref, reloaded), "params reflect training since start (sanity)")
    try:
      fake_grads(worker, names[0])
      worker.optim_step({"learning_rate": 1e-4}, names[0])
      check(True, "optim_step after disk reload works")
    except Exception as e:
      check(False, f"optim_step after disk reload raised: {e}")

  def p50(v):
    return sorted(v)[len(v) // 2] if v else float("nan")

  summary = {"event": "disk_baseline_summary", "mode": "disk_baseline", "model": BASE_MODEL,
             "rank": LORA_RANK, "flush_caches": FLUSH_CACHES,
             "cycles": CYCLES, "tenants": TENANTS, "swap_out_p50_ms": round(p50(outs), 1), "swap_in_p50_ms": round(p50(ins), 1),
             "n_out": len(outs), "n_in": len(ins), "failures": len(failures)}
  metrics.emit(**summary)
  print(f"[disk-baseline] SUMMARY {json.dumps(summary)}")
  print(f"[disk-baseline] {'PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
  return 0 if not failures else 1


if __name__ == "__main__":
  sys.exit(main())
