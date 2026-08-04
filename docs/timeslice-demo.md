# TimeSlice LoRA Swap Demo — Guide, Results, and Conclusions

## 1. What this demo did

Open-RL fine-tunes multiple LoRA adapters ("tenants") with RL on shared GPUs.
Stock Open-RL keeps **every** tenant's adapter weights *and* AdamW optimizer
state resident in GPU memory forever, and its vLLM sampler reloads adapter
weights from shared storage on every tenant switch.

This demo integrates the **TimeSlice Snapshot Agent** (from
[`llm-d-rl-time-slicing`](https://github.com/Edwinhr716/llm-d-rl-time-slicing/tree/gcr-backend-memory-allocation),
backend `BACKEND_GPU_CR_MEMORY_ADDRESSES`, built on
[GPU-CR](https://github.com/Edwinhr716/GPU-CR) v0.2.0 selective
checkpoint/restore) on **both sides of the RL loop**:

- **Sampler side** — vLLM runs with a single LoRA slot (`max_loras=1`).
  Instead of disk reloads, the slot's raw GPU bytes are snapshotted and
  restored through the node-local agent when tenants alternate
  (`server/timeslice_lora.py`, `server/timeslice_vllm_sampler.py`).
- **Trainer side** — at every tenant-batch boundary, the outgoing tenant's
  adapter weights + AdamW state are snapshotted and their **physical VRAM is
  released**; they are restored when that tenant's next batch arrives
  (`training/timeslice_tenant.py`, hook in
  `server/training_requests_processor.py`).

The end-to-end run (`scripts/timeslice_demo_driver.py`) drives two tenants
through 8 alternating rounds of the full Tinker-compatible REST loop
(forward_backward → optim_step → save_weights_for_sampler → sample) against
the real stack: gateway + redis queue + trainer worker (GPU 1) + vLLM sampler
(GPU 0), all on one GKE `g2-standard-24` node (2×L4). Each tenant trains
toward a distinct constant target token (420 vs 777), so adapter identity is
visible in temperature-0 outputs; determinism tripwires at the end verify
that swapping never corrupted anyone's weights.

**Result: 8 rounds × 2 tenants, zero failures.** Tenant-A locked onto its
target by round 2 and reproduced it bit-for-bit through six further rounds of
interleaved tenant-B training — i.e. its adapter + optimizer state survived
15 trainer swap-out/swap-in cycles and continuous sampler slot swaps intact.

## 2. Recreating the demo (no Cloud Build required)

### 2.0 Prerequisites

- A GKE **Standard** cluster (Autopilot is not supported: hostPath, hostPID,
  privileged pods, and node system config are required).
- `kubectl`, `docker`, and `gcloud` (only for cluster/registry auth).
- A docker registry you can push to (examples use Artifact Registry:
  `gcloud auth configure-docker asia-southeast1-docker.pkg.dev`).
- This branch (`timeslice-lora-swap`) of open-rl checked out.

Published images (skip any build step by using these):

| Component | Image |
|---|---|
| Snapshot agent | `asia-southeast1-docker.pkg.dev/edwinhernandez-gke-dev/time-slicing/llm-d-rl-time-slicing/snapshot-agent:hugetlbfs-v5` |
| Demo (sampler/trainer/gateway/driver) | `asia-southeast1-docker.pkg.dev/edwinhernandez-gke-dev/time-slicing/open-rl/timeslice-sampler:v8` |

### 2.1 GPU node pool with HugePages

GPU-CR stages GPU memory dumps in 2Mi HugePages. Each CUDA process reserves
27Gi at init (25Gi dump buffer + 2×1Gi staging, compile-time constants in
`vGPU-NVIDIA.so`), so the 60Gi pool fits exactly two GPU workloads per node.

```shell
cat > hugepages-config.yaml <<'EOF'
linuxConfig:
  hugepageConfig:
    hugepage_size2m: 30720   # 60Gi of 2Mi pages
EOF

# NOTE: g2-standard-24 REQUIRES exactly 2x L4 (count=1 is rejected).
# 300GB disk: the vLLM image is ~20GB unpacked; 100GB disks hit DiskPressure.
gcloud container node-pools create rl-hugepages-l4 \
  --cluster=<CLUSTER> --location=<REGION> --node-locations=<ZONE> \
  --machine-type=g2-standard-24 \
  --accelerator=type=nvidia-l4,count=2,gpu-driver-version=latest \
  --num-nodes=1 --disk-size=300 \
  --system-config-from-file=hugepages-config.yaml
```

### 2.2 Deploy the Snapshot Agent

The complete manifest is included at
`k8s/deploy/timeslice-demo/snapshot-agent.yaml`. Before applying, edit:
- `nodeSelector` → your node pool name,
- the image → your copy of `snapshot-agent:hugetlbfs-v5`.

```shell
kubectl create namespace timeslice-system
# ServiceAccount + RBAC come from the llm-d-rl-time-slicing helm chart
# (deploy/snapshot-agent on branch gcr-backend-memory-allocation); the
# DaemonSet below replaces the chart's daemonset template.
kubectl apply -f k8s/deploy/timeslice-demo/snapshot-agent.yaml
```

What this manifest encodes (all learned the hard way — see §5):
- An init container that **mounts hugetlbfs at `/var/tmp/huge-ckpt` on the
  host**. GPU-CR does not pass `MAP_HUGETLB`; without a real hugetlbfs mount
  every "hugepage" dump silently degrades to boot-disk page cache and the
  pool sits 100% unused.
- `mountPropagation: HostToContainer` so the agent sees that mount.
- `SNAPSHOT_DIR=/var/snapshots` backed by tmpfs (`/dev/shm/gcr-snapshots`):
  hugetlbfs has no `write(2)`, so snapshot copies cannot live next to the
  dumps; tmpfs keeps the restore path at RAM speed.
- `GPU_CR_COPY_HOST_FILE=0`: the 2GB `-host` DMA staging file is scratch and
  copying it dominated swap latency (verified restore works without it).
- 12Gi memory limit: tmpfs snapshot writes are charged to the **agent's**
  cgroup (~2GB per parked trainer tenant).

If you need to rebuild the agent image (requires the
`gcr-backend-memory-allocation` branch of `llm-d-rl-time-slicing` plus the
hugetlbfs patch set: `/proc/<pid>/maps` id fallback, mmap-write restore
copies, stale-artifact GC, 120s op timeout, FAULTED auto-recovery):

```shell
cd llm-d-rl-time-slicing
docker build -f docker/snapshot-agent/Dockerfile -t <YOUR_REGISTRY>/snapshot-agent:hugetlbfs-v5 .
docker push <YOUR_REGISTRY>/snapshot-agent:hugetlbfs-v5
```

### 2.3 Build the demo image locally

One image serves the sampler (default entrypoint), the trainer, the gateway,
and the driver. It is based on `vllm/vllm-openai:v0.22.0` — the combination
validated end-to-end with GPU-CR — and bakes in the prebuilt
`vGPU-NVIDIA.so` preloader and the `timeslice` gRPC client.

```shell
cd open-rl   # this branch
docker build -f src/server/Dockerfile.timeslice -t <YOUR_REGISTRY>/timeslice-sampler:v8 .
docker push <YOUR_REGISTRY>/timeslice-sampler:v8
```

### 2.4 Deploy the stack

Edit the image references in `k8s/deploy/timeslice-demo/stack.yaml` and
`driver-job.yaml` if you pushed your own, then:

```shell
kubectl apply -f k8s/deploy/timeslice-demo/stack.yaml
kubectl get pods -n openrl-demo -w   # redis, gateway, sampler, trainer
```

The sampler and trainer land on the hugepages node (one L4 each); redis,
gateway, and the driver run on any CPU node. Trainer and sampler share
adapters via hostPath `/var/tmp/open-rl` (a node-local stand-in for
Filestore — both GPU roles are on the same node).

### 2.5 THE CLEAN-SLATE PROCEDURE (required before every run)

Dead GPU-CR dump files pin their full ~27Gi hugepage **reservations** (they
attach to the file inode, not the process). Recycling GPU pods without
wiping the files starves the next GPU-CR init (`mmap with hugepages failed:
Cannot allocate memory`). Before every demo run:

```shell
kubectl scale deploy/timeslice-sampler deploy/trainer-worker -n openrl-demo --replicas=0
sleep 25
# privileged pod on the GPU node:
#   rm -rf /var/tmp/huge-ckpt/* /dev/shm/gcr-snapshots/* \
#          /var/tmp/open-rl/peft/* /var/tmp/open-rl/metrics/*
# then verify: grep HugePages_Free /proc/meminfo  -> must equal HugePages_Total
kubectl scale deploy/timeslice-sampler deploy/trainer-worker -n openrl-demo --replicas=1
kubectl rollout status deploy/timeslice-sampler deploy/trainer-worker -n openrl-demo --timeout=600s
```

### 2.6 Run: snapshot mode, then baseline

```shell
# Snapshot mode (TIMESLICE flags are "1" in stack.yaml)
kubectl apply -f k8s/deploy/timeslice-demo/driver-job.yaml
kubectl logs -f job/timeslice-demo-driver -n openrl-demo | grep '\[driver\]'

# Baseline (vLLM disk reloads, all tenants stay resident on the trainer)
kubectl delete job timeslice-demo-driver -n openrl-demo
kubectl set env deploy/timeslice-sampler TIMESLICE_ENABLED=0 -n openrl-demo
kubectl set env deploy/trainer-worker TIMESLICE_TRAINER_ENABLED=0 -n openrl-demo
# ... repeat the clean-slate procedure (2.5), then re-apply the driver job.
```

Driver knobs (env on the Job): `ROUNDS` (default 8), `LEARNING_RATE`
(default 1e-3 — hotter than ~1e-2 NaNs the tiny model), `BASE_MODEL`.

## 3. Viewing the results

**Visual report:** `docs/timeslice-demo-report.html` is a self-contained page
(no external assets, light/dark aware) with the stat tiles, swap-latency
chart, per-round snapshot-vs-baseline comparisons, the VRAM-freed chart, and
the token-convergence table — open it directly in a browser, serve it with
`python3 -m http.server -d docs`, or host it anywhere static. Regenerate it
from a new run's data with `scripts/timeslice_visual_report.py` (dataset
format: `docs/data/timeslice-run-<date>.json`).

1. **Driver log** — the run's narrative: per-round `train_round` / `sample`
   wall times, the sampled tokens (watch tenant-A converge to `420`),
   tripwire checks, and the final `PASSED`/`FAILED` verdict:
   ```shell
   kubectl logs job/timeslice-demo-driver -n openrl-demo | grep '\[driver\]'
   ```
2. **Swap metrics (jsonl)** — every swap event with wall time, agent-side
   elapsed time, and VRAM deltas, written to the shared volume:
   ```shell
   kubectl exec -n openrl-demo deploy/trainer-worker -- \
     cat /mnt/open-rl/metrics/trainer.jsonl /mnt/open-rl/metrics/sampler.jsonl > run.jsonl
   python3 scripts/timeslice_report.py run.jsonl     # p50/p95 table + VRAM
   ```
3. **Agent-side timings** — per-operation breakdown (file copy vs GPU work):
   ```shell
   kubectl logs -n timeslice-system -l app.kubernetes.io/name=snapshot-agent | grep took
   ```
4. **GPU dashboard** — Cloud Monitoring → "Open-RL: Accelerator Performance"
   (`dev/monitoring/apply_dashboard.sh <PROJECT>` deploys it): trainer vs
   sampler duty cycle and GPU memory, per container.
5. **Hugepages proof** — on the node, `grep Huge /proc/meminfo`:
   `HugePages_Free` drops below `HugePages_Total` while workloads run (it
   stayed at 100% free before the hugetlbfs fix).

## 4. Measured results & conclusions

Per-round numbers, Qwen2.5-0.5B, rank-16 LoRA, NVIDIA L4 (driver + metrics
from the 2026-08-04 runs; snapshot mode 8 full rounds, baseline 6):

| Metric | Snapshot mode | Baseline (all resident) |
|---|---|---|
| Sampler request incl. adapter switch | ~3.4–3.6s | ~2.3–2.4s |
| Trainer round (fb + optim + save) | ~10.8s | ~8.0s |
| Trainer tenant switch | 2865ms p50 | ~0 (all resident) |
| Sampler restore / snapshot op | 388ms / 737ms p50 | n/a (disk reload) |
| **Physical VRAM freed per parked tenant** | **~2.07GB (15/15 swaps)** | **0** |

**The resource-constrained comparison** (when VRAM *cannot* hold every
tenant, the real product scenario — measured with
`scripts/test_trainer_disk_baseline.py`, no GPU-CR env taxes):

| Trainer tenant switch strategy | Round-trip p50 | Data moved | VRAM back |
|---|---|---|---|
| Stay resident (unconstrained only) | 0 | 0 | 0 |
| Disk round-trip (`save_state`+`load_from_state`) | **1440ms** (790 out / 651 in) | 101MB serialized | tenant's real footprint |
| Snapshot agent swap (GPU-CR) | 2865ms | ~2GB (2MB blocks) | ~2.07GB (incl. block bloat) |

**The scale test — where the verdict flips** (Qwen3-4B-Instruct-2507,
rank 64, ~1502 regions, same L4; disk numbers cache-honest via
`FLUSH_CACHES=1`, i.e. `sync` after save + `drop_caches` before load):

| Strategy at 4B/rank-64 | Round-trip p50 | Notes |
|---|---|---|
| Disk round-trip | **18.1s** (9.4s out / 8.7s in) | scales with real bytes × storage speed + serialization + PEFT rebuild |
| Snapshot agent swap | **5.3s** (3.6s out / 1.7s in) | **3.4x faster**; frees 3.8GB VRAM/tenant; 1512 tensors bitwise-verified; block amplification naturally shrinks to ~1.6x at rank 64 |

The crossover is real and measured: disk multiplexing wins at toy scale
(~2x), the snapshot agent wins at production-like state sizes (~3.4x on
node-local disk — network filesystems widen it), and the VRAM story holds
at both scales.

**Conclusions:**

1. **The mechanism is correct.** Raw GPU-memory snapshot/restore of live
   vLLM LoRA slots and live trainer adapter+optimizer state is bit-stable
   across many cycles — including `optim_step` *after* restore, GPU-CR's
   historically risky write-after-restore path.
2. **Under resource pressure, the disk path wins at this scale.** Against
   the honest constrained baseline (Open-RL's own `save_state` +
   `load_from_state` eviction), the snapshot switch is ~2x slower today —
   because it moves ~20x more bytes than the real state. The unconstrained
   "all resident" comparison flatters neither direction: it answers a
   scenario where no multiplexing is needed at all.
3. **Block granularity is the whole ballgame.** GPU-CR checkpoints whole
   2MB-aligned allocations: 144 sampler regions ⇒ ~288MB moved for ~6MB of
   rank-16 weights; ~1000 trainer tensors ⇒ ~2GB per swap for 101MB of
   serializable state. The required no-caching-allocator setting also
   inflates the *resident* footprint (~330MB of packed tensors become ~2GB
   of blocks), so part of the "VRAM freed" is bloat the mechanism itself
   introduced. Upstream sub-block dumps would flip both the latency and the
   footprint comparisons; the snapshot path's structural advantages —
   RAM-speed staging, zero serialization/PEFT-rebuild cost, address
   stability — grow with model scale and slower shared storage
   (Filestore/NFS), where the disk path degrades and the snapshot path does
   not. Secondary tax either way: GPU-CR hard-requires
   `CUDA_LAUNCH_BLOCKING=1` + eager mode, slowing training and generation
   regardless of swapping.
4. **Operational discipline is part of the system.** Hugepage reservations
   attach to dump *files*, snapshot stores must be tenant-keyed and
   garbage-collected, and every workload restart needs the clean-slate
   procedure. The agent hardening added for this demo (op timeouts, FAULTED
   auto-recovery, stale-artifact GC) is what made iterating possible.

**Known open issues** (tracked in the demo plan):
- Tenant-B degenerated to NaN weights only when trainer swapping was on
  (healthy in baseline). Suspected interaction between real backward-pass
  gradients and the untracked-tensor filter (`GPU_CR_MAX_TRACKED_ADDR`,
  which excludes tensors allocated outside GPU-CR's hooked VMM window —
  checkpointing those crashes the workload). Needs a Gate-B variant driven
  by real `forward_backward` gradients to isolate.
- Upstream: gateway 500s when training metrics contain NaN; GPU-CR crashes
  on untracked regions instead of skipping; 2MB block granularity.
