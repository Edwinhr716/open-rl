"""LoRA slot swapping through the TimeSlice Snapshot Agent.

vLLM is run with max_loras=1, so a single GPU slot holds the active adapter.
Instead of letting vLLM reload adapter weights from shared storage on every
tenant switch, this manager snapshots the slot's raw GPU memory through the
node-local Snapshot Agent (BACKEND_GPU_CR_MEMORY_ADDRESSES / GPU-CR selective
checkpoint) and restores another tenant's bytes in place.

Swap semantics (validated by llm-d-rl-time-slicing's test_lora_swap_max1.py):

- The slot's stacked LoRA buffers (lora_a_stacked / lora_b_stacked) are
  preallocated by vLLM, so their device addresses are static across adapter
  loads. We discover them once, in-worker, after the first adapter load.
- Saving the resident adapter is snapshot THEN restore of the same group:
  GPU-CR's selective checkpoint releases the physical pages backing the
  regions (virtual addresses preserved), so an immediate restore is required
  to remap them before the slot is touched again. Skipping it causes an MMU
  fault (Xid 31) on the next write.
- Restoring a previously saved adapter overwrites the slot bytes while vLLM's
  metadata still names the resident adapter: requests must therefore be
  issued with the *carrier* LoRARequest (whatever vLLM believes is loaded),
  not the logical adapter's identity.

Every swap appends a metrics row (jsonl) so snapshot-mode and baseline runs
can be compared directly.
"""

import hashlib
import json
import os
import re
import threading
import time

METRICS_PATH = os.getenv("TIMESLICE_METRICS_PATH", "/tmp/timeslice-metrics.jsonl")


def _now_ms() -> float:
  return time.monotonic() * 1000.0


def lora_int_id(lora_id: str) -> int:
  """Stable 32-bit positive id, same scheme as vllm_sampler.py."""
  return int(hashlib.md5(lora_id.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1


def _tenant_key(lora_id: str) -> str:
  """Collapse a per-save session ref (tinker://<uuid>/sampler_weights/...)
  to its stable tenant id. Sessions change on every weight save; groups must
  not, or the snapshot store grows by one full slot dump per round."""
  if lora_id.startswith("tinker://"):
    parts = lora_id[len("tinker://") :].split("/")
    if len(parts) >= 3 and parts[1] == "sampler_weights":
      return parts[0]
  return lora_id


def _group_name(lora_id: str) -> str:
  """Snapshot group: one per TENANT, overwritten in place on each save.
  Freshness is tracked separately per session id (the _saved set), so a
  stale group is never restored — a new session disk-loads and re-snapshots
  over the tenant's group."""
  safe = re.sub(r"[^A-Za-z0-9._-]", "-", _tenant_key(lora_id))
  return f"lora-{safe[-80:]}"


def get_slot0_addresses_from_worker(model):
  """Runs inside the vLLM worker via apply_model: emit 'pid:hexaddr:size'
  targets for slot 0 of every stacked LoRA buffer.

  Ported from llm-d-rl-time-slicing testing-artifacts/test_lora_swap_max1.py.
  """
  import os as _os

  import torch
  from vllm.lora.layers import BaseLayerWithLoRA

  targets = []
  pid = _os.getpid()

  def emit(t):
    size = t.element_size() * t.nelement()
    if size > 0:
      targets.append(f"{pid}:{hex(t.data_ptr())}:{size}")

  for _name, module in model.named_modules():
    if not isinstance(module, BaseLayerWithLoRA):
      continue
    a, b = getattr(module, "lora_a_stacked", None), getattr(module, "lora_b_stacked", None)
    if a is None or b is None:
      continue
    if isinstance(a, tuple):
      for i in range(len(a)):
        emit(a[i][0])
        emit(b[i][0])
    elif isinstance(a, torch.Tensor):
      emit(a[0])
      emit(b[0])
  return targets


class MetricsWriter:
  def __init__(self, path: str = METRICS_PATH):
    self._path = path
    self._lock = threading.Lock()

  def emit(self, **row):
    row.setdefault("ts", time.time())
    with self._lock:
      d = os.path.dirname(self._path)
      if d:
        os.makedirs(d, exist_ok=True)
      with open(self._path, "a") as f:
        f.write(json.dumps(row) + "\n")


class TimesliceLoraManager:
  """Tracks slot residency and drives the Snapshot Agent on tenant switches.

  Callers must serialize calls (the sampler holds one lock around
  ensure_resident + generate); the manager itself is not re-entrant.
  """

  def __init__(self, enabled: bool | None = None, agent_endpoint: str | None = None, job_id: str | None = None, metrics: MetricsWriter | None = None):
    self.enabled = enabled if enabled is not None else os.getenv("TIMESLICE_ENABLED", "0") == "1"
    self.job_id = job_id or os.getenv("JOB_ID", "openrl-sampler")
    self.metrics = metrics or MetricsWriter()

    self._llm = None
    self._client = None
    self._backend = None
    self._slot_targets: list[str] | None = None

    self.resident_id: str | None = None       # logical adapter whose bytes are in the slot
    self.carrier: object | None = None        # LoRARequest vLLM believes is loaded
    self._saved: set[str] = set()             # adapter ids with a valid snapshot group
    self._last_id: str | None = None          # for baseline switch accounting

    if self.enabled:
      from timeslice.snapshot_agent import snapshot_agent_pb2
      from timeslice.snapshot_agent.client import SnapshotAgentClient

      endpoint = agent_endpoint or os.environ["AGENT_ENDPOINT"]
      self._client = SnapshotAgentClient(endpoint=endpoint)
      self._client.check_health()
      self._backend = snapshot_agent_pb2.BACKEND_GPU_CR_MEMORY_ADDRESSES
      print(f"[timeslice] connected to snapshot agent at {endpoint}")

  def attach_engine(self, llm):
    """Attach the sync vllm.LLM instance (needed for apply_model)."""
    self._llm = llm

  # -- internals ----------------------------------------------------------

  def _discover_slot(self):
    if self._slot_targets is None:
      self._slot_targets = self._llm.llm_engine.apply_model(get_slot0_addresses_from_worker)[0]
      print(f"[timeslice] slot0 addrs discovered ({len(self._slot_targets)} regions)")
      self.metrics.emit(event="discover", regions=len(self._slot_targets))

  def _agent_op(self, op: str, lora_id: str) -> dict:
    fn = self._client.snapshot_and_wait if op == "snapshot" else self._client.restore_and_wait
    t0 = _now_ms()
    resp = fn(job_id=self.job_id, group=_group_name(lora_id), backend=self._backend, memory_addresses=self._slot_targets, poll_interval_sec=0.05)
    wall = _now_ms() - t0
    if resp.status != "OPERATION_STATUS_COMPLETE":
      raise RuntimeError(f"{op}({lora_id}) failed: {resp.status} {resp.error}")
    stats = {"wall_ms": round(wall, 1), "agent_elapsed_ms": resp.elapsed_ms, "storage_bytes": resp.storage_bytes}
    self.metrics.emit(event=op, lora_id=lora_id, mode="snapshot", **stats)
    return stats

  def _save_resident(self):
    """Snapshot the resident adapter, then immediately restore it to remap
    the physical pages GPU-CR released (see module docstring)."""
    self._agent_op("snapshot", self.resident_id)
    self._agent_op("restore", self.resident_id)
    self._saved.add(self.resident_id)

  # -- public API ----------------------------------------------------------

  def ensure_resident(self, lora_id: str, lora_path: str, make_request):
    """Return the LoRARequest to issue for this generate call.

    make_request(lora_id, int_id, path) constructs a LoRARequest (passed in
    to keep this module import-safe without vllm).
    """
    switched = lora_id != self._last_id and self._last_id is not None
    self._last_id = lora_id

    if not self.enabled:
      # Baseline: vLLM's own LRU does the work; we only account switches.
      if switched:
        self.metrics.emit(event="switch", lora_id=lora_id, mode="baseline")
      return make_request(lora_id, lora_int_id(lora_id), lora_path)

    if self.resident_id == lora_id:
      return self.carrier

    t0 = _now_ms()
    if self.resident_id is None:
      # First adapter: plain disk load; discovery + save happen lazily later.
      self.carrier = make_request(lora_id, lora_int_id(lora_id), lora_path)
      self.resident_id = lora_id
      self.metrics.emit(event="first_load", lora_id=lora_id, mode="snapshot")
      return self.carrier

    self._discover_slot()
    if self.resident_id not in self._saved:
      self._save_resident()

    if lora_id in self._saved:
      # Hijack: land saved bytes under the carrier's identity.
      self._agent_op("restore", lora_id)
      self.resident_id = lora_id
      self.metrics.emit(event="switch", lora_id=lora_id, mode="snapshot", path="restore", wall_ms=round(_now_ms() - t0, 1))
      return self.carrier

    # Unknown adapter: let vLLM disk-load it into the (remapped) slot.
    # Older sessions of the same tenant become unrestorable the moment this
    # session is snapshotted over the tenant's group — forget them.
    tkey = _tenant_key(lora_id)
    self._saved = {s for s in self._saved if _tenant_key(s) != tkey}
    self.carrier = make_request(lora_id, lora_int_id(lora_id), lora_path)
    self.resident_id = lora_id
    self.metrics.emit(event="switch", lora_id=lora_id, mode="snapshot", path="disk_load", wall_ms=round(_now_ms() - t0, 1))
    return self.carrier

  def invalidate(self, lora_id: str):
    """Forget a saved snapshot (e.g. the trainer published new weights under
    a fresh session id; old groups for the same tenant become garbage)."""
    self._saved.discard(lora_id)
    if self.resident_id == lora_id:
      self.resident_id = None
