import logging
import os
import shlex
import subprocess
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class CheckpointRestorer(Protocol):
  def checkpoint(self, pid: int) -> None:
    pass

  def restore(self, pid: int) -> None:
    pass


class CudaCheckpointRestorer:
  def __init__(self, cuda_checkpoint_bin: str | None = None, timeout_ms: int | None = None):
    self.cuda_checkpoint_bin = cuda_checkpoint_bin or os.getenv("CUDA_CHECKPOINT_BIN", "cuda-checkpoint")
    self.timeout_ms = timeout_ms

  def checkpoint(self, pid: int) -> None:
    start = time.perf_counter()
    logger.info("checkpoint pid=%s", pid)
    lock_args = ["--action", "lock", "--pid", str(pid)]
    if self.timeout_ms is not None:
      lock_args.extend(["--timeout", str(self.timeout_ms)])

    self.run_cuda_checkpoint(lock_args)
    self.run_cuda_checkpoint(["--action", "checkpoint", "--pid", str(pid)])
    logger.info("checkpoint pid=%s took %.0f ms", pid, (time.perf_counter() - start) * 1000)

  def restore(self, pid: int) -> None:
    start = time.perf_counter()
    logger.info("restore pid=%s", pid)
    self.run_cuda_checkpoint(["--action", "restore", "--pid", str(pid)])
    self.run_cuda_checkpoint(["--action", "unlock", "--pid", str(pid)])
    logger.info("restore pid=%s took %.0f ms", pid, (time.perf_counter() - start) * 1000)

  def run_cuda_checkpoint(self, args: list[str]) -> None:
    full_argv = [self.cuda_checkpoint_bin, *args]
    result = subprocess.run(full_argv, capture_output=True, check=False, text=True)
    if result.returncode != 0:
      stderr = result.stderr.strip()
      stdout = result.stdout.strip()
      detail = stderr or stdout or f"exit code {result.returncode}"
      rendered_argv = " ".join(shlex.quote(arg) for arg in full_argv)
      raise RuntimeError(f"{rendered_argv} failed: {detail}")


class DirectMemoryRestorer:
  """Full-process park/resume through the timeslice Go snapshot-agent's
  direct_memory (GPU-CR) backend.

  E13 measured this against cuda-checkpoint on the same GPU: park p50 533ms
  vs 4427ms (8.3x), resume 510ms vs 1630ms (3.2x), 0/40 determinism
  mismatches. The speed comes with deployment requirements cuda-checkpoint
  does not have:

  - the Go agent must run with --feature-gates=DirectMemoryBackend=true;
  - every worker process must run under the GPU-CR vGPU preloader
    (LD_PRELOAD=vGPU-NVIDIA.so) with PYTORCH_NO_CUDA_MEMORY_CACHING=1 and
    CUDA_LAUNCH_BLOCKING=1 — a process started without the preloader
    cannot be parked by this backend;
  - the dump store (hugetlbfs mount) must be shared between the agent and
    the workers, with a hugepages allocation covering the parked VRAM.

  Each worker pid gets its own agent-side job (one park/resume state
  machine per process). This relies on the agent's standalone mode, which
  registers unknown job ids on first use; in k8s mode job state comes from
  the pod watcher and per-pid job ids would never leave IDLE.
  """

  def __init__(
    self,
    endpoint: str | None = None,
    job_prefix: str | None = None,
    precondition_retries: int = 12,
    precondition_delay_s: float = 5.0,
  ):
    from timeslice.snapshot_agent import SnapshotAgentClient, direct_memory_config

    self._direct_memory_config = direct_memory_config
    self.endpoint = endpoint or os.environ["AGENT_ENDPOINT"]
    self.job_prefix = job_prefix or os.getenv("JOB_ID", "openrl-park")
    self.precondition_retries = precondition_retries
    self.precondition_delay_s = precondition_delay_s
    self.client = SnapshotAgentClient(endpoint=self.endpoint)
    # Fails fast on an agent without the DirectMemoryBackend gate or with an
    # unresolvable cr_client, instead of on the first RELEASE.
    self.client.check_health("direct-memory")
    logger.info("direct-memory backend healthy at %s", self.endpoint)

  def _job_id(self, pid: int) -> str:
    return f"{self.job_prefix}-{pid}"

  def _config(self, pid: int):
    return self._direct_memory_config([pid])

  def checkpoint(self, pid: int) -> None:
    start = time.perf_counter()
    logger.info("checkpoint pid=%s", pid)
    # The agent marks a job RUNNING from GPU activity on its own cadence; a
    # snapshot issued before it has seen this pid's CUDA context (or against
    # stale SAVED state from a previous run of the same job id) gets
    # FAILED_PRECONDITION. Retry those; surface everything else.
    last: Exception | None = None
    for _ in range(self.precondition_retries):
      try:
        resp = self.client.snapshot_and_wait(
          job_id=self._job_id(pid), backend_config=self._config(pid), poll_interval_sec=0.05
        )
        break
      except Exception as exc:
        if "FAILED_PRECONDITION" not in str(exc):
          raise
        last = exc
        time.sleep(self.precondition_delay_s)
    else:
      raise RuntimeError(f"snapshot pid={pid} never left FAILED_PRECONDITION") from last
    self._check_complete(resp, "snapshot", pid)
    logger.info("checkpoint pid=%s took %.0f ms", pid, (time.perf_counter() - start) * 1000)

  def restore(self, pid: int) -> None:
    start = time.perf_counter()
    logger.info("restore pid=%s", pid)
    resp = self.client.restore_and_wait(
      job_id=self._job_id(pid), backend_config=self._config(pid), poll_interval_sec=0.05
    )
    self._check_complete(resp, "restore", pid)
    logger.info("restore pid=%s took %.0f ms", pid, (time.perf_counter() - start) * 1000)

  def _check_complete(self, resp, op: str, pid: int) -> None:
    if resp.status != "OPERATION_STATUS_COMPLETE":
      raise RuntimeError(f"{op} pid={pid} failed: {resp.status} {resp.error}")
