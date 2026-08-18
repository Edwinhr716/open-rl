# Snapshot Agent

The snapshot agent is a small process-level GPU residency primitive.

It exposes four commands over a Unix socket:

- `REGISTER(pid)` records a worker process.
- `ACQUIRE(pid)` grants that process the right to touch CUDA.
- `RELEASE(pid)` checkpoints that process before another process can acquire CUDA.
- `UNREGISTER(pid)` removes the process registration.

Today every successful `RELEASE` checkpoints the process. This is simple and
conservative, but it is slow because even a single run pays checkpoint cost after
each acquire window.

## Park backends

`RELEASE`/`ACQUIRE` park and resume through one of two backends, selected by
`--park-backend` (or `PARK_BACKEND`):

- `direct_memory` (default): full-process park through the timeslice Go
  snapshot-agent's GPU-CR backend. E13 measured park p50 533ms vs 4427ms and
  resume p50 510ms vs 1630ms against cuda-checkpoint on the same GPU
  (8.3x/3.2x), with 0/40 determinism mismatches. Requires:
  - `AGENT_ENDPOINT` pointing at a Go snapshot-agent running in standalone
    mode with `--feature-gates=DirectMemoryBackend=true`;
  - worker processes started under the GPU-CR vGPU preloader
    (`LD_PRELOAD=vGPU-NVIDIA.so`, `PYTORCH_NO_CUDA_MEMORY_CACHING=1`,
    `CUDA_LAUNCH_BLOCKING=1`);
  - the GPU-CR dump store (hugetlbfs mount) shared between the Go agent and
    the workers.
- `cuda`: the previous behavior — shell out to the `cuda-checkpoint` binary
  (`CUDA_CHECKPOINT_BIN`). No agent or preloader needed; slower, and device
  memory is copied into the worker's host RAM (size the memory limit for the
  full VRAM working set).
