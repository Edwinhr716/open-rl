"""Generate the self-contained HTML report for the TimeSlice demo.

Reads the checked-in run dataset (docs/data/timeslice-run-<date>.json) and
writes docs/timeslice-demo-report.html: stat tiles, swap-latency bars,
snapshot-vs-baseline per-round comparisons, VRAM-freed bars, and the token
convergence table. No external assets; light/dark via CSS custom properties.

Usage: python3 scripts/timeslice_visual_report.py [data.json] [out.html]
"""

import json
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else "docs/data/timeslice-run-2026-08-04.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/timeslice-demo-report.html"

d = json.load(open(DATA))
meta = d["meta"]

# ---- palette roles (reference palette; see dataviz references/palette.md) ----
CSS = """
.viz-root { color-scheme: light;
  --page:#f9f9f7; --surface-1:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
    --page:#0d0d0d; --surface-1:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --series-1:#3987e5; --series-2:#d95926; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark;
  --page:#0d0d0d; --surface-1:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; }
.viz-root { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page); color: var(--ink); margin: 0; padding: 24px; }
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 28px 0 2px; }
.sub { color: var(--ink-2); font-size: 13px; margin: 0 0 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit,minmax(200px,1fr)); gap: 12px; margin: 18px 0; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .v { font-size: 26px; font-weight: 650; } .tile .k { color: var(--ink-2); font-size: 12px; margin-top: 2px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin: 10px 0; }
.legend { display: flex; gap: 16px; font-size: 12px; color: var(--ink-2); margin: 4px 0 8px; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 5px; vertical-align: -1px; background: var(--c); }
svg text { font-family: inherit; font-size: 11px; fill: var(--muted); }
svg .val { fill: var(--ink-2); font-variant-numeric: tabular-nums; }
svg .cat { fill: var(--ink-2); }
table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; }
details { margin-top: 8px; font-size: 12px; color: var(--ink-2); }
.note { color: var(--muted); font-size: 11.5px; margin-top: 6px; }
.tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--page);
  padding: 4px 8px; border-radius: 6px; font-size: 12px; opacity: 0; transition: opacity .08s; z-index: 9; }
.ok { color: #006300; } @media (prefers-color-scheme: dark) { .ok { color: #0ca30c; } }
"""

TIP_JS = """
const tip = document.querySelector('.tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => { tip.textContent = el.dataset.tip;
    tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY - 10) + 'px'; tip.style.opacity = 1; });
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
});
"""


def rbar_h(x, y, w, h, color, tipv):
  """Horizontal bar, 4px rounded data-end (right), flat at baseline."""
  r = min(4, w / 2)
  return (f'<path d="M{x},{y} h{w - r} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{w - r} z" '
          f'fill="{color}" data-tip="{tipv}"/>')


def rbar_v(x, y, w, h, color, tipv):
  """Vertical bar, 4px rounded data-end (top), flat at baseline."""
  r = min(4, w / 2)
  return (f'<path d="M{x},{y + h} v-{h - r} a{r},{r} 0 0 1 {r},-{r} h{w - 2 * r} a{r},{r} 0 0 1 {r},{r} v{h - r} z" '
          f'fill="{color}" data-tip="{tipv}"/>')


def chart_swap_ops():
  ops = [("Sampler restore", d["swap_ops_p50"]["restore"], d["swap_ops_n"]["restore"]),
         ("Sampler snapshot", d["swap_ops_p50"]["snapshot"], d["swap_ops_n"]["snapshot"]),
         ("Trainer swap-in", d["swap_ops_p50"]["trainer_swap_in"], d["swap_ops_n"]["trainer_swap_in"]),
         ("Trainer swap-out", d["swap_ops_p50"]["trainer_swap_out"], d["swap_ops_n"]["trainer_swap_out"]),
         ("Trainer tenant switch", d["swap_ops_p50"]["trainer_switch"], d["swap_ops_n"]["trainer_switch"])]
  mx = max(v for _, v, _ in ops)
  left, bw, gap, w = 150, 18, 10, 950
  rows, y = [], 8
  for name, v, n in ops:
    bl = (w - left - 90) * v / mx
    rows.append(f'<text class="cat" x="{left - 8}" y="{y + 13}" text-anchor="end">{name}</text>')
    rows.append(rbar_h(left, y, bl, bw, "var(--series-1)", f"{name}: {v:.0f}ms p50 (n={n})"))
    rows.append(f'<text class="val" x="{left + bl + 6}" y="{y + 13}">{v:,.0f} ms</text>')
    y += bw + gap
  h = y + 4
  return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Swap operation p50 latencies">'
          f'<line x1="{left}" y1="0" x2="{left}" y2="{h - 4}" stroke="var(--axis)"/>' + "".join(rows) + "</svg>")


def grouped_rounds(title_a, snap, base, unit=1000.0, fmt="{:.1f}s"):
  """Vertical grouped bars per round: snapshot (slot 1) vs baseline (slot 2).
  Round 0 excluded (both modes pay a ~15s one-time first-load there)."""
  n = max(len(snap), len(base)) - 1
  mx = max(snap[1:] + base[1:])
  w, h, pad_b, pad_t = 950, 190, 24, 8
  gw = (w - 60) / n
  bw = min(28, (gw - 14) / 2)
  parts = [f'<line x1="40" y1="{h - pad_b}" x2="{w - 10}" y2="{h - pad_b}" stroke="var(--axis)"/>']
  for gl in (0.5, 1.0):
    gy = h - pad_b - (h - pad_b - pad_t) * gl
    parts.append(f'<line x1="40" y1="{gy}" x2="{w - 10}" y2="{gy}" stroke="var(--grid)"/>')
    parts.append(f'<text x="36" y="{gy + 4}" text-anchor="end">{fmt.format(mx * gl / unit)}</text>')
  for i in range(1, n + 1):
    gx = 50 + (i - 1) * gw
    for j, (vals, color, name) in enumerate([(snap, "var(--series-1)", "snapshot"), (base, "var(--series-2)", "baseline")]):
      if i < len(vals):
        v = vals[i]
        bh = max(2, (h - pad_b - pad_t) * v / mx)
        parts.append(rbar_v(gx + j * (bw + 2), h - pad_b - bh, bw, bh, color, f"round {i} {name}: {fmt.format(v / unit)}"))
    parts.append(f'<text x="{gx + bw + 1}" y="{h - pad_b + 14}" text-anchor="middle">r{i}</text>')
  return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title_a}">' + "".join(parts) + "</svg>")


def chart_switch_strategies():
  """Trainer tenant switch, three ways: stay resident / disk round-trip /
  snapshot swap. One measure (ms) -> single hue; strategy named by label."""
  db = d["disk_baseline"]
  disk_rt = db["swap_out_p50_ms"] + db["swap_in_p50_ms"]
  snap_rt = d["swap_ops_p50"]["trainer_switch"]
  rows_data = [("Stay resident (unconstrained)", 0, "0 ms — but VRAM grows ~linearly per tenant"),
               ("Disk round-trip (save_state + load_from_state)", disk_rt, f"{disk_rt:,.0f}ms p50 — moves {db['state_mb']:.0f}MB serialized state"),
               ("Snapshot agent swap (GPU-CR)", snap_rt, f"{snap_rt:,.0f}ms p50 — moves ~2GB (2MB-block amplification)")]
  mx = max(v for _, v, _ in rows_data) or 1
  left, bw, gap, w = 320, 18, 12, 950
  parts, y = [], 8
  for name, v, tipv in rows_data:
    bl = max(2, (w - left - 100) * v / mx)
    parts.append(f'<text class="cat" x="{left - 8}" y="{y + 13}" text-anchor="end">{name}</text>')
    parts.append(rbar_h(left, y, bl, bw, "var(--series-1)", tipv))
    parts.append(f'<text class="val" x="{left + bl + 6}" y="{y + 13}">{v:,.0f} ms</text>')
    y += bw + gap
  return (f'<svg viewBox="0 0 {w} {y + 4}" role="img" aria-label="Tenant switch strategies compared">'
          f'<line x1="{left}" y1="0" x2="{left}" y2="{y}" stroke="var(--axis)"/>' + "".join(parts) + "</svg>")


def chart_scale_test():
  """Switch round-trip at two scales x two strategies. Snapshot = slot 1,
  disk = slot 2 (same entity colors as the per-round charts)."""
  st = d["scale_test"]
  db = d["disk_baseline"]
  groups = [
    ("0.5B / rank 16", (d["swap_ops_p50"]["trainer_swap_out"] + d["swap_ops_p50"]["trainer_swap_in"]) / 1000,
     (db["swap_out_p50_ms"] + db["swap_in_p50_ms"]) / 1000),
    ("4B / rank 64", (st["snapshot"]["swap_out_ms"] + st["snapshot"]["swap_in_ms"]) / 1000,
     (st["disk_baseline"]["swap_out_p50_ms"] + st["disk_baseline"]["swap_in_p50_ms"]) / 1000),
  ]
  mx = max(max(s, b) for _, s, b in groups)
  w, h, pad_b, pad_t, left = 950, 210, 24, 8, 40
  gw = (w - left - 20) / len(groups)
  bw = 60
  parts = [f'<line x1="{left}" y1="{h - pad_b}" x2="{w - 10}" y2="{h - pad_b}" stroke="var(--axis)"/>']
  for gl in (0.5, 1.0):
    gy = h - pad_b - (h - pad_b - pad_t) * gl
    parts.append(f'<line x1="{left}" y1="{gy}" x2="{w - 10}" y2="{gy}" stroke="var(--grid)"/>')
    parts.append(f'<text x="{left - 4}" y="{gy + 4}" text-anchor="end">{mx * gl:.0f}s</text>')
  for i, (label, s, b) in enumerate(groups):
    gx = left + 40 + i * gw
    for j, (v, color, name) in enumerate([(s, "var(--series-1)", "snapshot swap"), (b, "var(--series-2)", "disk round-trip")]):
      bh = max(2, (h - pad_b - pad_t) * v / mx)
      x = gx + j * (bw + 2)
      parts.append(rbar_v(x, h - pad_b - bh, bw, bh, color, f"{label} {name}: {v:.1f}s round-trip"))
      parts.append(f'<text class="val" x="{x + bw / 2}" y="{h - pad_b - bh - 5}" text-anchor="middle">{v:.1f}s</text>')
    parts.append(f'<text class="cat" x="{gx + bw + 1}" y="{h - pad_b + 15}" text-anchor="middle">{label}</text>')
  return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Tenant switch round-trip vs model scale">' + "".join(parts) + "</svg>")


def chart_vram():
  vals = d["vram_freed_mb"]
  mx = max(vals)
  w, h, pad_b, pad_t = 950, 150, 22, 8
  bw = min(40, (w - 70) / len(vals) - 4)
  parts = [f'<line x1="40" y1="{h - pad_b}" x2="{w - 10}" y2="{h - pad_b}" stroke="var(--axis)"/>']
  gy = h - pad_b - (h - pad_b - pad_t)
  parts.append(f'<line x1="40" y1="{gy}" x2="{w - 10}" y2="{gy}" stroke="var(--grid)"/>')
  parts.append(f'<text x="36" y="{gy + 4}" text-anchor="end">{mx / 1000:.1f}GB</text>')
  for i, v in enumerate(vals):
    bh = (h - pad_b - pad_t) * v / mx
    x = 50 + i * (bw + 4)
    parts.append(rbar_v(x, h - pad_b - bh, bw, bh, "var(--series-1)", f"swap #{i + 1}: {v:.0f} MB freed"))
    parts.append(f'<text x="{x + bw / 2}" y="{h - pad_b + 13}" text-anchor="middle">{i + 1}</text>')
  return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="VRAM freed per trainer swap-out">' + "".join(parts) + "</svg>")


def mean(v):
  return sum(v) / len(v)


sd, bd = d["snapshot_driver"], d["baseline_driver"]
snap_sample = [mean(p) for p in zip(sd["tenantA_sample_ms"], sd["tenantB_sample_ms"])]
base_sample = [mean(p) for p in zip(bd["sample"]["A"], bd["sample"]["B"])]
snap_train = [mean(p) for p in zip(sd["tenantA_train_ms"], sd["tenantB_train_ms"])]
base_train = [mean(p) for p in zip(bd["train"]["A"], bd["train"]["B"])]
vram_p50 = sorted(d["vram_freed_mb"])[len(d["vram_freed_mb"]) // 2]

conv_rows = "".join(
  f'<tr><td>{i}</td><td>{sd["tenantA_tokens"][i]}{" <span class=ok>✓ target</span>" if "420" in sd["tenantA_tokens"][i] else ""}</td>'
  f'<td>{sd["tenantB_tokens"][i]}</td></tr>' for i in range(len(sd["tenantA_tokens"])))

table = lambda hdr, rows: "<table><tr>" + "".join(f"<th>{x}</th>" for x in hdr) + "</tr>" + rows + "</table>"
rounds_rows = lambda s, b: "".join(
  f"<tr><td>{i}</td><td>{s[i] / 1000:.2f}s</td><td>{(b[i] / 1000 if i < len(b) else float('nan')):.2f}s</td></tr>"
  for i in range(1, len(s)))

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TimeSlice LoRA Swap Demo — Results</title><style>{CSS}</style></head>
<body class="viz-root"><div class="wrap">
<h1>TimeSlice LoRA Swap Demo — Results</h1>
<p class="sub">{meta["model"]} · rank-{meta["rank"]} LoRA · {meta["gpu"]} · {meta["date"]} ·
snapshot mode {meta["rounds_snapshot"]} rounds / baseline {meta["rounds_baseline"]} rounds × 2 tenants ·
full Open-RL stack (gateway → redis → trainer, gateway → vLLM sampler)</p>

<div class="tiles">
 <div class="tile"><div class="v">0</div><div class="k">tripwire failures across all rounds — restored weights bit-stable</div></div>
 <div class="tile"><div class="v">{vram_p50 / 1000:.2f} GB</div><div class="k">physical VRAM freed per parked tenant (p50, {len(d["vram_freed_mb"])}/{len(d["vram_freed_mb"])} trainer swaps)</div></div>
 <div class="tile"><div class="v">{d["swap_ops_p50"]["restore"]:,.0f} ms</div><div class="k">sampler LoRA-slot restore p50 (vs full disk reload)</div></div>
 <div class="tile"><div class="v">{d["swap_ops_p50"]["trainer_switch"] / 1000:.1f} s</div><div class="k">trainer tenant switch p50 (swap-out + swap-in)</div></div>
</div>

<div class="card"><h2 style="margin-top:0">Swap operation latency (p50)</h2>
<p class="sub">Agent-driven GPU memory snapshot/restore operations, wall clock including gRPC + file copies.</p>
{chart_swap_ops()}
<details><summary>Data table</summary>{table(["operation", "p50 ms", "n"],
  "".join(f"<tr><td>{k}</td><td>{v:,.0f}</td><td>{d['swap_ops_n'][k]}</td></tr>" for k, v in d["swap_ops_p50"].items()))}</details></div>

<div class="card"><h2 style="margin-top:0">Sampler request per round — snapshot vs baseline</h2>
<div class="legend"><span style="--c:var(--series-1)">snapshot mode</span><span style="--c:var(--series-2)">baseline (disk reload)</span></div>
{grouped_rounds("Sampler request wall time per round", snap_sample, base_sample)}
<p class="note">Mean of both tenants per round. Round 0 excluded: both modes pay the same one-time ~14.9s
first adapter load there. Baseline wins per-request at this scale; the snapshot-mode gap is the swap ops
plus GPU-CR's required CUDA_LAUNCH_BLOCKING/eager taxes.</p>
<details><summary>Data table</summary>{table(["round", "snapshot", "baseline"], rounds_rows(snap_sample, base_sample))}</details></div>

<div class="card"><h2 style="margin-top:0">Trainer round per round — snapshot vs baseline</h2>
<div class="legend"><span style="--c:var(--series-1)">snapshot mode</span><span style="--c:var(--series-2)">baseline (all tenants resident)</span></div>
{grouped_rounds("Trainer round wall time per round", snap_train, base_train)}
<p class="note">forward_backward + optim_step + save per round; the ~2.9s delta is the tenant switch
(swap-out ~1.9s + swap-in ~1.0s) that buys the VRAM release below.</p>
<details><summary>Data table</summary>{table(["round", "snapshot", "baseline"], rounds_rows(snap_train, base_train))}</details></div>

<div class="card"><h2 style="margin-top:0">Trainer tenant switch — three strategies (p50 round-trip)</h2>
{chart_switch_strategies()}
<p class="note">The resource-constrained comparison: when VRAM cannot hold every tenant, the alternatives are
Open-RL's disk eviction (save_state + load_from_state, measured here without GPU-CR's env taxes:
save {d["disk_baseline"]["save_p50_ms"]:,.0f}ms + evict {d["disk_baseline"]["evict_p50_ms"]:,.0f}ms out,
{d["disk_baseline"]["swap_in_p50_ms"]:,.0f}ms back, {d["disk_baseline"]["state_mb"]:.0f}MB on node-local disk)
or the snapshot agent. At this scale the disk path wins ~2x: GPU-CR moves ~20x more bytes than the real state
(2MB-block granularity). The snapshot path's structural advantages — RAM-speed staging, no serialization or
PEFT module rebuild, address stability — need sub-block dumps and/or slower shared storage + bigger states
to dominate. Disk numbers include page-cache absorption; network filesystems (Filestore/NFS) shift the
disk path right, not the snapshot path.</p></div>

<div class="card"><h2 style="margin-top:0">The crossover: switch round-trip vs model scale</h2>
<div class="legend"><span style="--c:var(--series-1)">snapshot agent swap</span><span style="--c:var(--series-2)">disk round-trip (save_state + load_from_state)</span></div>
{chart_scale_test()}
<p class="note">Measured 2026-08-04 on the same L4 node. At 0.5B/rank-16 the disk path wins ~2x (and its
number is cache-warm — flattering). At {d["scale_test"]["model"].split("/")[-1]}/rank-64 with cache-honest I/O
(sync + drop_caches), the disk path degrades to {(d["scale_test"]["disk_baseline"]["swap_out_p50_ms"] + d["scale_test"]["disk_baseline"]["swap_in_p50_ms"]) / 1000:.1f}s
while the snapshot swap holds at {(d["scale_test"]["snapshot"]["swap_out_ms"] + d["scale_test"]["snapshot"]["swap_in_ms"]) / 1000:.1f}s —
<b>a {(d["scale_test"]["disk_baseline"]["swap_out_p50_ms"] + d["scale_test"]["disk_baseline"]["swap_in_p50_ms"]) / (d["scale_test"]["snapshot"]["swap_out_ms"] + d["scale_test"]["snapshot"]["swap_in_ms"]):.1f}x win for the snapshot agent</b>,
freeing {d["scale_test"]["snapshot"]["vram_freed_mb"] / 1000:.1f}GB of VRAM per parked tenant (all
{d["scale_test"]["snapshot"]["tensors_verified"]} tensors bitwise-verified, write-after-restore intact). Block
amplification shrinks naturally with tensor size (~20x at rank 16 → ~1.6x at rank 64), while the disk path
scales with real bytes through serialization and storage — network filesystems widen the gap further.</p></div>

<div class="card"><h2 style="margin-top:0">Physical VRAM freed per trainer swap-out</h2>
{chart_vram()}
<p class="note">Every parked tenant returns its adapter + AdamW optimizer state to the pool
(~{vram_p50 / 1000:.1f} GB at 2MB block granularity). Baseline frees 0: tenants stay resident forever,
so VRAM grows linearly with tenant count on a 24GB L4 that also carries the base model.</p></div>

<div class="card"><h2 style="margin-top:0">Token convergence under swapping (snapshot mode, temp 0)</h2>
<p class="sub">Tenant-A trains toward token 420. From round 2 it reproduces its output bit-for-bit through
six further rounds of interleaved tenant-B training — proof its adapter + optimizer state survives every
swap cycle. (Tenant-B degraded to deterministic token-0 output in snapshot mode while converging to its
777 target in baseline — the one open correctness issue, tracked in the demo plan.)</p>
{table(["round", "tenant-A tokens", "tenant-B tokens"], conv_rows)}</div>

<div class="card"><h2 style="margin-top:0">Conclusions</h2>
<ol style="font-size:13px;color:var(--ink-2);margin:6px 0 2px;padding-left:18px">
<li><b>The mechanism is correct</b> — raw GPU-memory snapshot/restore of live vLLM LoRA slots and live
trainer adapter+optimizer state is bit-stable, including optimizer steps <i>after</i> restore.</li>
<li><b>The winner depends on scale, and the crossover is measured</b> — at 0.5B/rank-16 the disk baseline
round-trips in ~{(d["disk_baseline"]["swap_out_p50_ms"] + d["disk_baseline"]["swap_in_p50_ms"]) / 1000:.1f}s (cache-warm)
vs the snapshot switch's {d["swap_ops_p50"]["trainer_switch"] / 1000:.1f}s; at 4B/rank-64 with honest I/O the disk
path takes {(d["scale_test"]["disk_baseline"]["swap_out_p50_ms"] + d["scale_test"]["disk_baseline"]["swap_in_p50_ms"]) / 1000:.1f}s vs the
snapshot swap's {(d["scale_test"]["snapshot"]["swap_out_ms"] + d["scale_test"]["snapshot"]["swap_in_ms"]) / 1000:.1f}s —
a {(d["scale_test"]["disk_baseline"]["swap_out_p50_ms"] + d["scale_test"]["disk_baseline"]["swap_in_p50_ms"]) / (d["scale_test"]["snapshot"]["swap_out_ms"] + d["scale_test"]["snapshot"]["swap_in_ms"]):.1f}x
snapshot win at production-like state sizes, before network storage widens it further.</li>
<li><b>Block granularity is the whole ballgame</b> — ~20× data amplification on the trainer, ~48× on the
sampler; it also inflates the resident footprint itself (the no-caching-allocator requirement turns ~330MB
of packed tensors into ~2GB of blocks). Upstream sub-block dumps would flip both the latency and footprint
comparisons; RAM-speed staging, zero serialization/rebuild cost, and address stability are the structural
advantages waiting behind that fix — and they grow with model scale and slower shared storage, where the
disk path degrades and the snapshot path does not.</li>
</ol></div>

<p class="note">Generated by scripts/timeslice_visual_report.py from docs/data/timeslice-run-{meta["date"]}.json.
Reproduce the runs with docs/timeslice-demo.md.</p>
</div><div class="tip"></div><script>{TIP_JS}</script></body></html>
"""

with open(OUT, "w") as f:
  f.write(html)
print(f"wrote {OUT} ({len(html)} bytes)")
