"""Open-RL vLLM sampler with TimeSlice LoRA slot swapping.

Drop-in alternative to server.vllm_sampler with the same HTTP contract
(POST /generate, GET /healthz) but backed by the sync vllm.LLM engine and a
TimesliceLoraManager. The sync engine matches the configuration validated
end-to-end with GPU-CR (VLLM_ENABLE_V1_MULTIPROCESSING=0, enforce_eager,
apply_model address discovery); AsyncLLMEngine is unproven with the
preloader, so requests are serialized instead — acceptable for RL rollout
workloads where tenants alternate in batches.

Env:
  BASE_MODEL                  model to serve (required)
  TIMESLICE_ENABLED           1 = swap adapters via snapshot agent; 0 = vLLM disk reload (baseline)
  AGENT_ENDPOINT              snapshot agent gRPC endpoint (host:9001)
  TIMESLICE_SELFTEST          1 = run the A/B hijack self-test and exit instead of serving
  VLLM_MAX_LORA_RANK          default 16
  VLLM_GPU_MEMORY_UTILIZATION default 0.5
  VLLM_MAX_MODEL_LEN          default 2048
  TIMESLICE_METRICS_PATH      metrics jsonl (default /tmp/timeslice-metrics.jsonl)
"""

import json
import os
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI, Request

from server.timeslice_lora import METRICS_PATH, TimesliceLoraManager

_llm = None
_manager: TimesliceLoraManager | None = None
_gen_lock = threading.Lock()


def _build_engine():
  global _llm, _manager
  from vllm import LLM

  model_name = os.environ["BASE_MODEL"]
  _llm = LLM(
    model=model_name,
    enable_lora=True,
    max_loras=1,  # single slot: switching is the thing being demonstrated
    max_lora_rank=int(os.getenv("VLLM_MAX_LORA_RANK", "16")),
    max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
    gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5")),
    enforce_eager=True,  # required for GPU-CR
    disable_custom_all_reduce=True,  # required for GPU-CR
    enable_prefix_caching=False,
  )
  _manager = TimesliceLoraManager()
  _manager.attach_engine(_llm)
  print(f"[timeslice-sampler] engine ready model={model_name} timeslice={'ON' if _manager.enabled else 'OFF (baseline)'}")


def _generate(prompt, sampling_params, lora_id=None, lora_path=None):
  """Serialized generate through the manager. prompt: text or token-id dict."""
  from vllm.lora.request import LoRARequest

  with _gen_lock:
    lora_request = None
    if lora_id and lora_path:
      t0 = time.monotonic()
      lora_request = _manager.ensure_resident(lora_id, lora_path, lambda lid, iid, p: LoRARequest(lid, iid, p))
      swap_ms = (time.monotonic() - t0) * 1000.0
    outputs = _llm.generate([prompt], sampling_params, lora_request=lora_request, use_tqdm=False)
    if lora_id and lora_path:
      _manager.metrics.emit(event="generate", lora_id=lora_id, mode="snapshot" if _manager.enabled else "baseline", ensure_ms=round(swap_ms, 1))
    return outputs[0]


app = FastAPI(title="Open-RL TimeSlice vLLM Sampler")


@app.on_event("startup")
def _startup():
  _build_engine()


@app.get("/healthz")
def healthz():
  return {"status": "ok", "timeslice": _manager.enabled if _manager else None}


@app.post("/generate")
async def generate(req: Request):
  from vllm import SamplingParams

  try:
    data = await req.json()
    prompt_token_ids = data.get("prompt_token_ids")
    prompt = {"prompt_token_ids": prompt_token_ids} if prompt_token_ids else data.get("prompt", "")
    sampling_params = SamplingParams(
      n=data.get("num_samples", 1),
      temperature=data.get("temperature", 1.0),
      max_tokens=data.get("max_tokens", 20),
      stop_token_ids=data.get("stop"),
      top_p=data.get("top_p", 1.0),
      top_k=data.get("top_k", -1),
      logprobs=1,
    )
    final = _generate(prompt, sampling_params, data.get("lora_id"), data.get("lora_path"))

    import math

    sequences_out = []
    for output in final.outputs:
      token_ids = list(output.token_ids)
      logprobs = []
      if output.logprobs:
        for idx, tl in enumerate(output.logprobs):
          lp = tl[token_ids[idx]].logprob if tl and token_ids[idx] in tl else -9999.0
          logprobs.append(lp if math.isfinite(lp) else -9999.0)
      sequences_out.append({"tokens": token_ids, "logprobs": logprobs, "stop_reason": output.finish_reason, "text": output.text})
    return {"sequences": sequences_out, "prompt_logprobs": None}
  except Exception as e:
    import traceback

    traceback.print_exc()
    return {"type": "RequestFailedResponse", "error_message": f"TimeSlice Sampler Error: {str(e)}"}


# ---------------------------------------------------------------------------
# Self-test: A/B hijack correctness + swap latency, exercised through the
# exact serving code path (_generate). Runs on the cluster as a one-shot Job.
# ---------------------------------------------------------------------------


def _make_dummy_adapters():
  """Two deterministic dummy PEFT adapters (B = 10x A), written to /tmp.
  Ported from llm-d-rl-time-slicing testing-artifacts/test_lora_swap_max1.py."""
  import torch
  from vllm.lora.layers import BaseLayerWithLoRA

  def gen(model, lora_dir, mult):
    os.makedirs(lora_dir, exist_ok=True)
    target_modules, state_dict = [], {}
    rank = int(os.getenv("VLLM_MAX_LORA_RANK", "16"))
    for name, module in model.named_modules():
      if not isinstance(module, BaseLayerWithLoRA):
        continue
      if name.endswith(".qkv_proj"):
        subs = [name[:-8] + "q_proj", name[:-8] + "v_proj"]
      elif name.endswith(".gate_up_proj"):
        subs = [name[:-12] + "gate_proj", name[:-12] + "up_proj"]
      else:
        subs = [name]
      for sub in subs:
        suffix = sub.split(".")[-1]
        if suffix not in target_modules:
          target_modules.append(suffix)
        in_dim = getattr(module, "input_size", None)
        out_dim = getattr(module, "output_size", None)
        if in_dim is None or out_dim is None:
          base = getattr(module, "base_layer", None)
          if base is not None:
            in_dim = getattr(base, "input_size", getattr(base, "in_features", None))
            out_dim = getattr(base, "output_size", getattr(base, "out_features", None))
        if not in_dim or not out_dim:
          continue
        if name.endswith(".qkv_proj"):
          out_dim //= 3
        elif name.endswith(".gate_up_proj"):
          out_dim //= 2
        state_dict[f"base_model.model.{sub}.lora_A.weight"] = torch.ones(rank, in_dim) * 0.01 * mult
        state_dict[f"base_model.model.{sub}.lora_B.weight"] = torch.ones(out_dim, rank) * 0.01 * mult
    config = {
      "r": rank, "lora_alpha": rank * 2, "target_modules": target_modules,
      "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM",
      "peft_type": "LORA", "base_model_name_or_path": "dummy",
    }
    with open(os.path.join(lora_dir, "adapter_config.json"), "w") as f:
      json.dump(config, f)
    torch.save(state_dict, os.path.join(lora_dir, "adapter_model.bin"))

  _llm.llm_engine.apply_model(lambda m: (gen(m, "/tmp/lora_A", 1.0), gen(m, "/tmp/lora_B", 10.0)) and None)


def selftest(rounds: int) -> int:
  from vllm import SamplingParams

  _build_engine()
  _make_dummy_adapters()
  sp = SamplingParams(temperature=0.0, max_tokens=10)
  adapters = {"tenant-A": "/tmp/lora_A", "tenant-B": "/tmp/lora_B"}
  expected: dict[str, str] = {}
  switch_ms: list[float] = []
  failures = 0

  for r in range(rounds):
    for lora_id, path in adapters.items():
      t0 = time.monotonic()
      out = _generate("Who are you?", sp, lora_id, path)
      wall = (time.monotonic() - t0) * 1000.0
      text = out.outputs[0].text
      if lora_id not in expected:
        expected[lora_id] = text
        print(f"[selftest] round {r} {lora_id}: initial output {text!r}")
      elif text != expected[lora_id]:
        failures += 1
        print(f"[selftest] round {r} {lora_id}: FAILURE — got {text!r}, expected {expected[lora_id]!r}")
      else:
        switch_ms.append(wall)
        print(f"[selftest] round {r} {lora_id}: OK ({wall:.0f}ms incl. swap)")

  mode = "snapshot" if _manager.enabled else "baseline"
  if switch_ms:
    switch_ms.sort()
    p50 = switch_ms[len(switch_ms) // 2]
    p95 = switch_ms[int(len(switch_ms) * 0.95) - 1] if len(switch_ms) > 1 else switch_ms[-1]
    summary = {"event": "selftest_summary", "mode": mode, "rounds": rounds, "switches": len(switch_ms), "failures": failures, "p50_ms": round(p50, 1), "p95_ms": round(p95, 1)}
    _manager.metrics.emit(**summary)
    print(f"[selftest] SUMMARY {json.dumps(summary)}")
  print(f"[selftest] metrics at {METRICS_PATH}:")
  with open(METRICS_PATH) as f:
    sys.stdout.write(f.read())
  print(f"[selftest] {'PASSED' if failures == 0 else 'FAILED'} mode={mode}")
  return 0 if failures == 0 else 1


if __name__ == "__main__":
  if os.getenv("TIMESLICE_SELFTEST", "0") == "1":
    sys.exit(selftest(rounds=int(os.getenv("TIMESLICE_SELFTEST_ROUNDS", "5"))))
  uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SAMPLER_PORT", "8001")))
