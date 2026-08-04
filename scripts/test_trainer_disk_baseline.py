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
  save_ms = (time.monotonic() - t0) * 1000

  free_before = torch.cuda.mem_get_info()[0]
  t0 = time.monotonic()
  other = next(t for t in worker.adapter_states if t != tenant)
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
  t0 = time.monotonic()
  worker.load_from_state(tenant, path, restore_optimizer=True)
  torch.cuda.synchronize()
  row = {"wall_ms": round((time.monotonic() - t0) * 1000, 1)}
  metrics.emit(event="disk_swap_in", lora_id=tenant, mode="disk_baseline", **row)
  return row


def main() -> int:
  print(f"[disk-baseline] loading {BASE_MODEL} + 2 tenants (rank 16), state root {STATE_ROOT}")
  os.makedirs(STATE_ROOT, exist_ok=True)
  worker = LoraTrainingWorker()
  cfg = LoraConfig(rank=16, seed=0)
  worker.create_model(BASE_MODEL, "tenant-A", cfg)
  worker.create_adapter("tenant-B", cfg)
  for tenant in ("tenant-A", "tenant-B"):
    fake_grads(worker, tenant)
    worker.optim_step({"learning_rate": 1e-4}, tenant)
  torch.cuda.synchronize()

  ref = worker.adapter_states["tenant-A"]["trainable_params"][0].detach().cpu().clone()

  outs, ins = [], []
  parked = None  # multiplex: at most one tenant parked on disk at a time
  for c in range(CYCLES):
    for tenant in ("tenant-A", "tenant-B"):
      t0 = time.monotonic()
      if parked == tenant:
        ins.append(swap_in_disk(worker, tenant)["wall_ms"])
        parked = None
      other = "tenant-B" if tenant == "tenant-A" else "tenant-A"
      if parked is None and other in worker.adapter_states:
        outs.append(swap_out_disk(worker, other)["wall_ms"])
        parked = other
      metrics.emit(event="disk_switch", lora_id=tenant, mode="disk_baseline",
                   wall_ms=round((time.monotonic() - t0) * 1000, 1))
      fake_grads(worker, tenant)
      worker.optim_step({"learning_rate": 1e-4}, tenant)
      print(f"[disk-baseline] cycle {c} {tenant}: active, optim_step OK")

  # Correctness: park A, reload, first param should differ from the ORIGINAL
  # reference only by its training (sanity: reload restores what was saved).
  if parked != "tenant-A" and "tenant-A" in worker.adapter_states:
    saved = worker.adapter_states["tenant-A"]["trainable_params"][0].detach().cpu().clone()
    swap_out_disk(worker, "tenant-A")
    swap_in_disk(worker, "tenant-A")
    reloaded = worker.adapter_states["tenant-A"]["trainable_params"][0].detach().cpu()
    check(torch.equal(saved, reloaded), "reloaded params identical to what was saved")
    check(not torch.equal(ref, reloaded), "params reflect training since start (sanity)")
    try:
      fake_grads(worker, "tenant-A")
      worker.optim_step({"learning_rate": 1e-4}, "tenant-A")
      check(True, "optim_step after disk reload works")
    except Exception as e:
      check(False, f"optim_step after disk reload raised: {e}")

  def p50(v):
    return sorted(v)[len(v) // 2] if v else float("nan")

  summary = {"event": "disk_baseline_summary", "mode": "disk_baseline", "model": BASE_MODEL,
             "cycles": CYCLES, "swap_out_p50_ms": round(p50(outs), 1), "swap_in_p50_ms": round(p50(ins), 1),
             "n_out": len(outs), "n_in": len(ins), "failures": len(failures)}
  metrics.emit(**summary)
  print(f"[disk-baseline] SUMMARY {json.dumps(summary)}")
  print(f"[disk-baseline] {'PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
  return 0 if not failures else 1


if __name__ == "__main__":
  sys.exit(main())
