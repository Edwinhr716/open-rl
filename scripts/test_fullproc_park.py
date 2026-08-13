"""Full-process park/resume gate: cuda-checkpoint vs GPU-CR direct_memory.

Starts a vLLM engine in-process, records a temperature-0 baseline completion
and steady-state generation throughput, then runs ROUNDS park/resume cycles
through the Go snapshot-agent using the backend selected by PARK_BACKEND
(``cuda`` or ``direct_memory``). Each cycle asserts:

  - park frees device memory (NVML, read out-of-band from the CUDA context);
  - resume restores it;
  - the post-resume temp-0 completion is bit-identical to the baseline.

The process must not touch CUDA between park and resume: both backends
freeze/evict this process's device state (cuda-checkpoint locks the CUDA
APIs; GPU-CR releases the dumped pages until restore).

Emits one JSON line per event to TIMESLICE_METRICS_PATH (or stdout) in the
same shape the timeslice report scripts consume. Exits 0 with
"[park-gate] PASSED" on success.

Env: PARK_BACKEND, ROUNDS, JOB_ID, AGENT_ENDPOINT, BASE_MODEL,
VLLM_GPU_MEMORY_UTILIZATION, ENFORCE_EAGER, TIMESLICE_METRICS_PATH.
"""

import json
import os
import statistics
import sys
import time

from timeslice.snapshot_agent.client import SnapshotAgentClient
from timeslice.snapshot_agent import snapshot_agent_pb2 as pb

PARK_BACKEND = os.environ.get("PARK_BACKEND", "cuda")
ROUNDS = int(os.environ.get("ROUNDS", "20"))
JOB_ID = os.environ.get("JOB_ID", f"park-gate-{PARK_BACKEND.replace('_', '-')}")
BASE_MODEL = os.environ.get("BASE_MODEL", "facebook/opt-125m")
GPU_UTIL = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.4"))
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"
METRICS_PATH = os.environ.get("TIMESLICE_METRICS_PATH", "")

PROMPT = "The capital of France is"
THROUGHPUT_PROMPTS = ["Write a short story about a robot." for _ in range(8)]


def emit(event, **fields):
    line = json.dumps({"ts": time.time(), "event": event, "backend": PARK_BACKEND, **fields})
    print(f"[metrics] {line}", flush=True)
    if METRICS_PATH:
        with open(METRICS_PATH, "a") as f:
            f.write(line + "\n")


def build_config(pid):
    target = pb.ProcessTarget(pids=[pid])
    if PARK_BACKEND == "direct_memory":
        return pb.BackendConfig(direct_memory=pb.DirectMemoryBackendConfig(explicit_target=target))
    if PARK_BACKEND == "cuda":
        return pb.BackendConfig(cuda=pb.CudaBackendConfig(explicit_target=target))
    raise SystemExit(f"unknown PARK_BACKEND {PARK_BACKEND!r}")


_NVML_SNIPPET = (
    "import pynvml,os;pynvml.nvmlInit();"
    "d=pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get('CUDA_VISIBLE_DEVICES','0')));"
    "print(pynvml.nvmlDeviceGetMemoryInfo(d).used//(1024*1024))"
)


def gpu_mem_used_mb():
    # Read VRAM from a fresh subprocess: while THIS process is parked, its
    # own driver calls (including NVML) are frozen by cuda-checkpoint, so an
    # in-process query would deadlock the gate.
    import subprocess

    out = subprocess.run(
        [sys.executable, "-c", _NVML_SNIPPET],
        capture_output=True, text=True, timeout=30, check=True,
        env={k: v for k, v in os.environ.items() if k != "LD_PRELOAD"},
    )
    return float(out.stdout.strip())


def wait_op(op, phase):
    status = getattr(op, "status", "")
    if "COMPLETE" not in str(status):
        raise SystemExit(f"[park-gate] FAILED: {phase} did not complete: status={status} error={getattr(op, 'error', '')}")


def main():
    from vllm import LLM, SamplingParams

    pid = os.getpid()  # host PID: the gate pod runs with hostPID: true
    print(f"[park-gate] backend={PARK_BACKEND} rounds={ROUNDS} pid={pid} model={BASE_MODEL}", flush=True)

    llm = LLM(model=BASE_MODEL, gpu_memory_utilization=GPU_UTIL, enforce_eager=ENFORCE_EAGER)
    greedy = SamplingParams(temperature=0.0, max_tokens=32)

    baseline = llm.generate([PROMPT], greedy)[0].outputs[0].text
    print(f"[park-gate] baseline completion: {baseline!r}", flush=True)

    # Steady-state throughput (the workload tax the arm pays while running).
    tp = SamplingParams(temperature=0.0, max_tokens=64)
    t0 = time.monotonic()
    outs = llm.generate(THROUGHPUT_PROMPTS, tp)
    gen_s = time.monotonic() - t0
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    toks_per_s = gen_tokens / gen_s
    emit("throughput", tokens=gen_tokens, seconds=round(gen_s, 3), toks_per_s=round(toks_per_s, 1))

    client = SnapshotAgentClient(endpoint=os.environ["AGENT_ENDPOINT"])
    health = client.check_health()
    print(f"[park-gate] agent health: {health.status}", flush=True)

    cfg = build_config(pid)
    vram_running = gpu_mem_used_mb()
    emit("vram_running", mb=round(vram_running))

    def snapshot_with_retry():
        # The agent's pod watcher marks the job RUNNING from GPU activity on
        # its own cadence; a snapshot issued before that (or against stale
        # state from a previous pod of the same job) gets FAILED_PRECONDITION.
        last = None
        for _ in range(12):
            try:
                return client.snapshot_and_wait(job_id=JOB_ID, backend_config=cfg, poll_interval_sec=0.05)
            except Exception as e:  # SnapshotAgentError wrapping FAILED_PRECONDITION
                if "FAILED_PRECONDITION" not in str(e):
                    raise
                last = e
                time.sleep(5)
        raise last

    park_ms, resume_ms, freed_mb, mismatches = [], [], [], 0
    for i in range(ROUNDS):
        t0 = time.monotonic()
        op = snapshot_with_retry()
        park = (time.monotonic() - t0) * 1000
        wait_op(op, f"park round {i}")

        time.sleep(1.0)  # parked dwell; NO CUDA calls in here
        vram_parked = gpu_mem_used_mb()
        freed = vram_running - vram_parked

        t0 = time.monotonic()
        op = client.restore_and_wait(job_id=JOB_ID, backend_config=cfg, poll_interval_sec=0.05)
        resume = (time.monotonic() - t0) * 1000
        wait_op(op, f"resume round {i}")

        text = llm.generate([PROMPT], greedy)[0].outputs[0].text
        match = text == baseline
        if not match:
            mismatches += 1
            print(f"[park-gate] round {i}: MISMATCH {text!r} != baseline", flush=True)

        park_ms.append(park)
        resume_ms.append(resume)
        freed_mb.append(freed)
        emit("cycle", round=i, park_ms=round(park), resume_ms=round(resume),
             vram_parked_mb=round(vram_parked), freed_mb=round(freed), match=match)

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    summary = {
        "rounds": ROUNDS,
        "park_ms_p50": round(statistics.median(park_ms)),
        "park_ms_p95": round(pct(park_ms, 0.95)),
        "resume_ms_p50": round(statistics.median(resume_ms)),
        "resume_ms_p95": round(pct(resume_ms, 0.95)),
        "freed_mb_p50": round(statistics.median(freed_mb)),
        "vram_running_mb": round(vram_running),
        "gen_toks_per_s": round(toks_per_s, 1),
        "mismatches": mismatches,
    }
    emit("park_gate_summary", **summary)

    if mismatches:
        print(f"[park-gate] FAILED: {mismatches}/{ROUNDS} determinism mismatches", flush=True)
        sys.exit(1)
    if statistics.median(freed_mb) < 500:
        print("[park-gate] FAILED: parking freed <500MB — backend did not release device memory", flush=True)
        sys.exit(1)
    print(f"[park-gate] PASSED {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
