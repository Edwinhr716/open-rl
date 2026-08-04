"""Phase 3 demo driver: two RL tenants alternating through the full Open-RL
stack (gateway -> redis queue -> trainer worker; gateway -> vLLM sampler),
exercising trainer-side tenant swaps and sampler-side LoRA slot swaps on
every round.

Speaks the gateway's Tinker-compatible REST API directly (no tinker SDK
dependency). Per tenant per round: forward_backward -> optim_step ->
save_weights_for_sampler -> asample. Ends with a determinism tripwire:
back-to-back temp-0 samples per tenant must match, and the two tenants'
outputs must differ.

Env: GATEWAY_URL (default http://gateway:8000), BASE_MODEL, ROUNDS, OUT_PATH.
"""

import json
import os
import sys
import time
import urllib.request

GATEWAY = os.getenv("GATEWAY_URL", "http://gateway:8000").rstrip("/")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
ROUNDS = int(os.getenv("ROUNDS", "5"))
OUT_PATH = os.getenv("OUT_PATH", "/tmp/driver-metrics.jsonl")

PROMPT_TOKENS = list(range(1, 17))
# Per-tenant target patterns: strong constant-token biases so the two
# adapters' temp-0 behaviors visibly diverge within a few optim steps —
# which is also what makes the cross-tenant tripwire meaningful. Keep LR
# below ~1e-2: hotter runs NaN the model within 2 steps, and the gateway
# crashes JSON-encoding NaN training metrics (upstream sanitization gap).
TENANT_TARGETS = {"tenant-A": [420] * 16, "tenant-B": [777] * 16}
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-3"))


def post(path: str, payload: dict) -> dict:
  req = urllib.request.Request(
    f"{GATEWAY}/api/v1/{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
  )
  with urllib.request.urlopen(req, timeout=180) as resp:
    return json.loads(resp.read())


def wait_future(request_id: str, timeout_s: float = 300.0, tolerate_500: bool = False) -> dict:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    try:
      result = post("retrieve_future", {"request_id": request_id})
    except Exception as e:  # gateway returns 400 while pending-with-error only
      # Known upstream gap: the gateway 500s when a training result carries
      # NaN metrics (json encoder). The op itself completed on the worker;
      # for train-phase futures we log and continue.
      if tolerate_500 and "500" in str(e):
        print(f"[driver] WARNING: future {request_id} returned 500 (NaN metrics?); continuing")
        return {"type": "tolerated_500"}
      raise RuntimeError(f"retrieve_future({request_id}) failed: {e}") from e
    if result and result.get("status") != "pending":
      return result
    time.sleep(0.5)
  raise TimeoutError(f"future {request_id} not ready after {timeout_s}s")


def emit(row: dict):
  row["ts"] = time.time()
  with open(OUT_PATH, "a") as f:
    f.write(json.dumps(row) + "\n")
  print(f"[driver] {json.dumps(row)}")


def make_datum(tenant: str) -> dict:
  targets = TENANT_TARGETS[tenant]
  return {
    "model_input": {"chunks": [{"tokens": PROMPT_TOKENS}]},
    "loss_fn_inputs": {
      "target_tokens": targets,
      "weights": [1.0] * len(targets),
    },
  }


def train_round(tenant: str, model_id: str, seq: int) -> str:
  t0 = time.monotonic()
  fb = post("forward_backward", {"model_id": model_id, "forward_backward_input": {"data": [make_datum(tenant)] * 4, "loss_fn": "cross_entropy"}})
  wait_future(fb["request_id"], tolerate_500=True)
  opt = post("optim_step", {"model_id": model_id, "adam_params": {"learning_rate": LEARNING_RATE, "grad_clip_norm": 1.0}})
  wait_future(opt["request_id"], tolerate_500=True)
  save = post("save_weights_for_sampler", {"model_id": model_id, "sampling_session_seq_id": seq})
  wait_future(save["request_id"])
  emit({"event": "train_round", "tenant": model_id[:8], "seq": seq, "wall_ms": round((time.monotonic() - t0) * 1000, 1)})
  return f"tinker://{model_id}/sampler_weights/sampler-{seq}"


def sample(session_ref: str, max_tokens: int = 12, tag: str = "") -> list[int]:
  t0 = time.monotonic()
  req = post(
    "asample",
    {
      "model_id": session_ref,
      "prompt": {"chunks": [{"tokens": PROMPT_TOKENS}]},
      "sampling_params": {"max_tokens": max_tokens, "temperature": 0.0},
      "num_samples": 1,
    },
  )
  result = wait_future(req["request_id"])
  if result.get("type") == "RequestFailedResponse":
    raise RuntimeError(f"sample failed: {result.get('error_message')}")
  tokens = result["sequences"][0]["tokens"]
  emit({"event": "sample", "tenant": session_ref.split("/")[2][:8] if "//" in session_ref else session_ref[:8], "tag": tag,
        "wall_ms": round((time.monotonic() - t0) * 1000, 1), "tokens": tokens[:6]})
  return tokens


def main() -> int:
  failures = []
  print(f"[driver] gateway={GATEWAY} base_model={BASE_MODEL} rounds={ROUNDS}")

  tenants = {}
  for name in ("tenant-A", "tenant-B"):
    resp = post("create_model", {"base_model": BASE_MODEL, "lora_config": {"rank": int(os.getenv("LORA_RANK", "16")), "seed": 0}})
    model_id = resp["request_id"]
    wait_future(model_id)
    tenants[name] = model_id
    print(f"[driver] {name} -> {model_id}")

  sessions = {}
  for r in range(ROUNDS):
    for name, model_id in tenants.items():
      sessions[name] = train_round(name, model_id, seq=r)
      sample(sessions[name], tag=f"round-{r}")

  # Determinism tripwire: temp-0 resample per tenant twice, then cross-check.
  finals = {}
  for name, ref in sessions.items():
    a = sample(ref, tag="tripwire-1")
    b = sample(ref, tag="tripwire-2")
    if a != b:
      failures.append(f"{name}: non-deterministic outputs across back-to-back samples (swap corruption?)")
    finals[name] = a
  if finals["tenant-A"] == finals["tenant-B"]:
    failures.append("tenant-A and tenant-B produced identical outputs (adapters not actually swapped?)")

  emit({"event": "driver_summary", "rounds": ROUNDS, "tenants": 2, "failures": failures})
  print(f"[driver] {'PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
  return 0 if not failures else 1


if __name__ == "__main__":
  sys.exit(main())
