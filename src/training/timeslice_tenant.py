"""Trainer-side tenant swapping through the TimeSlice Snapshot Agent.

Open-RL's LoRA trainer keeps every tenant's adapter weights AND AdamW state
(exp_avg/exp_avg_sq, ~4-5x the adapter size) resident on the GPU forever.
This manager evicts an inactive tenant's GPU tensors at the tenant-batch
boundary — GPU-CR selective checkpoint copies them to host and releases the
physical VRAM — and restores them when that tenant's next batch arrives.

Rules learned from code audit + Phase 0/1:
- Only swap a tenant AFTER its first optim_step: AdamW state materializes
  lazily on the first step(), and swapping params without their optimizer
  state would leave the optimizer referencing released memory.
- Snapshot then swap_in must use the SAME address list (the agent restores by
  pid:addr:size). Addresses are stable across set_adapter, but
  load_from_state reallocates the adapter — call forget() there.
- The snapshot slot (MemoryRegionsBackendConfig.snapshot_name) is one fixed
  name per tenant, overwritten on every swap_out: the trainer's bytes change
  every optim_step, so stale slots are useless and per-version slots would
  accumulate on the tmpfs store. The request's `group` field is orchestrator
  bookkeeping and does not name agent-side storage.
- Requires PYTORCH_NO_CUDA_MEMORY_CACHING=1 (GPU-CR tracks whole cudaMalloc
  blocks) and CUDA_LAUNCH_BLOCKING=1 (async launches race the physical
  release — hard requirement, verified in Phase 1).
"""

import os
import re
import time

import torch

from server.timeslice_lora import MetricsWriter


def _now_ms() -> float:
  return time.monotonic() * 1000.0


def _snapshot_name(model_id: str) -> str:
  return "trainer-" + re.sub(r"[^A-Za-z0-9._-]", "-", model_id)[-80:]


# The GPU-CR preloader's VMM hook maps device memory in a low virtual range
# (0x3xx.. in practice). Tensors occasionally land at host-heap-range
# addresses (allocated outside the hooked cudaMalloc path); checkpointing an
# untracked region crashes the workload's signal handler and hangs cr_client
# (observed in Phase 3). Filter those out: they stay resident on GPU —
# partial VRAM saving, full correctness.
GPU_CR_MAX_TRACKED_ADDR = int(os.getenv("GPU_CR_MAX_TRACKED_ADDR", str(0x100000000000)))


class TimesliceTenantManager:
  def __init__(self, enabled: bool | None = None, agent_endpoint: str | None = None, job_id: str | None = None, metrics: MetricsWriter | None = None):
    self.enabled = enabled if enabled is not None else os.getenv("TIMESLICE_TRAINER_ENABLED", "0") == "1"
    self.job_id = job_id or os.getenv("JOB_ID", "openrl-trainer")
    self.metrics = metrics or MetricsWriter()
    self.worker = None
    self._client = None
    self._swapped_out: dict[str, list[str]] = {}  # model_id -> address list used at snapshot time

    if self.enabled:
      from timeslice.snapshot_agent import SnapshotAgentClient

      endpoint = agent_endpoint or os.environ["AGENT_ENDPOINT"]
      self._client = SnapshotAgentClient(endpoint=endpoint)
      # Per-backend health: verifies the agent can resolve cr_client.
      self._client.check_health("memory-regions")
      print(f"[timeslice-trainer] connected to snapshot agent at {endpoint} (memory-regions healthy)")

  def attach_worker(self, worker):
    self.worker = worker

  def addresses_for(self, model_id: str) -> list[str] | None:
    """pid:hexaddr:size targets for a tenant's adapter params + AdamW state.
    Returns None if the tenant has no optimizer state yet (not swappable)."""
    state = self.worker.adapter_states.get(model_id)
    if not state:
      return None
    optimizer = state.get("optimizer")
    if optimizer is None:
      return None

    pid = os.getpid()
    tensors: list[torch.Tensor] = []
    for p in state["trainable_params"]:
      tensors.append(p.data)
      opt_state = optimizer.state.get(p, {})
      for key in ("exp_avg", "exp_avg_sq"):
        t = opt_state.get(key)
        if t is not None and t.is_cuda:
          tensors.append(t)
    if not tensors:
      return None
    targets, skipped = [], 0
    for t in tensors:
      if t.nelement() == 0:
        continue
      if t.data_ptr() >= GPU_CR_MAX_TRACKED_ADDR:
        skipped += 1
        continue
      targets.append(f"{pid}:{hex(t.data_ptr())}:{t.element_size() * t.nelement()}")
    if skipped:
      self.metrics.emit(event="trainer_untracked_tensors", lora_id=model_id, skipped=skipped, swappable=len(targets))
    return targets or None

  def _op(self, op: str, model_id: str, addresses: list[str]) -> dict:
    from timeslice.snapshot_agent import memory_regions_config

    fn = self._client.snapshot_and_wait if op == "snapshot" else self._client.restore_and_wait
    t0 = _now_ms()
    config = memory_regions_config(addresses, snapshot_name=_snapshot_name(model_id))
    resp = fn(job_id=self.job_id, group=_snapshot_name(model_id), backend_config=config, poll_interval_sec=0.05)
    wall = _now_ms() - t0
    if resp.status != "OPERATION_STATUS_COMPLETE":
      raise RuntimeError(f"trainer {op}({model_id}) failed: {resp.status} {resp.error}")
    return {"wall_ms": round(wall, 1), "agent_elapsed_ms": resp.elapsed_ms}

  def swap_out(self, model_id: str):
    """Snapshot an inactive tenant's tensors and release their physical VRAM."""
    if not self.enabled or model_id in self._swapped_out:
      return
    addresses = self.addresses_for(model_id)
    if not addresses:
      self.metrics.emit(event="trainer_swap_out_skipped", lora_id=model_id, reason="no optimizer state yet")
      return
    free_before = torch.cuda.mem_get_info()[0]
    stats = self._op("snapshot", model_id, addresses)
    torch.cuda.synchronize()
    free_after = torch.cuda.mem_get_info()[0]
    self._swapped_out[model_id] = addresses
    self.metrics.emit(event="trainer_swap_out", lora_id=model_id, mode="snapshot", regions=len(addresses),
                      vram_freed_mb=round((free_after - free_before) / 1e6, 1), **stats)

  def swap_in(self, model_id: str):
    """Restore a swapped-out tenant's tensors (remap + refill)."""
    if not self.enabled or model_id not in self._swapped_out:
      return
    addresses = self._swapped_out.pop(model_id)
    free_before = torch.cuda.mem_get_info()[0]
    stats = self._op("restore", model_id, addresses)
    torch.cuda.synchronize()
    free_after = torch.cuda.mem_get_info()[0]
    self.metrics.emit(event="trainer_swap_in", lora_id=model_id, mode="snapshot", regions=len(addresses),
                      vram_used_mb=round((free_before - free_after) / 1e6, 1), **stats)

  def switch_tenant(self, prev: str | None, next_id: str):
    """Called at the tenant-batch boundary in the requests processor."""
    if not self.enabled or prev == next_id:
      return
    t0 = _now_ms()
    # Restore the incoming tenant BEFORE checkpointing the outgoing one: the
    # per-PID staging dump file is shared across snapshot groups, and GPU-CR's
    # restore resolves offsets against the file's freshest extents — the
    # out-then-in order deterministically transplants the outgoing tenant's
    # bytes into a slice of the incoming tenant's optimizer blocks (negative
    # values in exp_avg_sq -> AdamW sqrt(neg) -> NaN). Costs both tenants
    # transiently resident during the switch.
    if os.getenv("TIMESLICE_SWAP_IN_FIRST", "1") == "1":
      self.swap_in(next_id)
      if prev is not None:
        self.swap_out(prev)
    else:
      if prev is not None:
        self.swap_out(prev)
      self.swap_in(next_id)
    self.metrics.emit(event="trainer_switch", lora_id=next_id, prev=prev, mode="snapshot", wall_ms=round(_now_ms() - t0, 1))

  def forget(self, model_id: str):
    """Tenant tensors were reallocated (load_from_state / create_adapter):
    prior addresses and snapshots are invalid."""
    self._swapped_out.pop(model_id, None)
