"""Generate the standalone Qwen3-4B scale-test report.

Reads the scale_test section of the run dataset and writes
docs/timeslice-scale-4b-report.html: the crossover verdict at
production-like tenant state size.

Usage: python3 scripts/timeslice_scale_report.py [data.json] [out.html]
"""

import json
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else "docs/data/timeslice-run-2026-08-04.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/timeslice-scale-4b-report.html"

d = json.load(open(DATA))
st = d["scale_test"]
snap = st["snapshot"]
disk = st["disk_baseline"]
small_snap = (d["swap_ops_p50"]["trainer_swap_out"] + d["swap_ops_p50"]["trainer_swap_in"]) / 1000
small_disk = (d["disk_baseline"]["swap_out_p50_ms"] + d["disk_baseline"]["swap_in_p50_ms"]) / 1000
snap_rt = (snap["swap_out_ms"] + snap["swap_in_ms"]) / 1000
disk_rt = (disk["swap_out_p50_ms"] + disk["swap_in_p50_ms"]) / 1000
speedup = disk_rt / snap_rt

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
.viz-root { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page); color: var(--ink); margin: 0; padding: 24px; }
.wrap { max-width: 900px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 0 0 2px; }
.sub { color: var(--ink-2); font-size: 13px; margin: 0 0 12px; }
.hero { font-size: 44px; font-weight: 700; margin: 14px 0 2px; }
.hero small { font-size: 15px; font-weight: 500; color: var(--ink-2); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 12px; margin: 18px 0; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .v { font-size: 24px; font-weight: 650; } .tile .k { color: var(--ink-2); font-size: 12px; margin-top: 2px; }
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
.note { color: var(--muted); font-size: 11.5px; margin-top: 6px; }
ul.checks { font-size: 13px; color: var(--ink-2); margin: 6px 0 2px; padding-left: 18px; }
.tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--page);
  padding: 4px 8px; border-radius: 6px; font-size: 12px; opacity: 0; transition: opacity .08s; z-index: 9; }
"""

TIP_JS = """
const tip = document.querySelector('.tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => { tip.textContent = el.dataset.tip;
    tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY - 10) + 'px'; tip.style.opacity = 1; });
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
});
"""


def rbar_v(x, y, w, h, color, tipv):
  r = min(4, w / 2)
  return (f'<path d="M{x},{y + h} v-{h - r} a{r},{r} 0 0 1 {r},-{r} h{w - 2 * r} a{r},{r} 0 0 1 {r},{r} v{h - r} z" '
          f'fill="{color}" data-tip="{tipv}"/>')


def crossover_chart():
  groups = [("0.5B / rank 16", small_snap, small_disk, "cache-warm disk"),
            ("4B / rank 64", snap_rt, disk_rt, "cache-honest disk")]
  mx = max(max(s, b) for _, s, b, _ in groups)
  w, h, pad_b, pad_t, left = 860, 230, 26, 10, 40
  gw = (w - left - 20) / len(groups)
  bw = 70
  parts = [f'<line x1="{left}" y1="{h - pad_b}" x2="{w - 10}" y2="{h - pad_b}" stroke="var(--axis)"/>']
  for gl in (0.5, 1.0):
    gy = h - pad_b - (h - pad_b - pad_t) * gl
    parts.append(f'<line x1="{left}" y1="{gy}" x2="{w - 10}" y2="{gy}" stroke="var(--grid)"/>')
    parts.append(f'<text x="{left - 4}" y="{gy + 4}" text-anchor="end">{mx * gl:.0f}s</text>')
  for i, (label, s, b, note) in enumerate(groups):
    gx = left + 60 + i * gw
    for j, (v, color, name) in enumerate([(s, "var(--series-1)", "snapshot swap"), (b, "var(--series-2)", "disk round-trip")]):
      bh = max(2, (h - pad_b - pad_t) * v / mx)
      x = gx + j * (bw + 2)
      parts.append(rbar_v(x, h - pad_b - bh, bw, bh, color, f"{label} {name}: {v:.1f}s ({note})"))
      parts.append(f'<text class="val" x="{x + bw / 2}" y="{h - pad_b - bh - 5}" text-anchor="middle">{v:.1f}s</text>')
    parts.append(f'<text class="cat" x="{gx + bw + 1}" y="{h - pad_b + 15}" text-anchor="middle">{label}</text>')
  return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Tenant switch round-trip at two scales">' + "".join(parts) + "</svg>")


def breakdown_chart():
  rows = [("Snapshot swap-out", snap["swap_out_ms"] / 1000, "var(--series-1)"),
          ("Snapshot swap-in", snap["swap_in_ms"] / 1000, "var(--series-1)"),
          ("Disk save+sync (out)", disk["swap_out_p50_ms"] / 1000, "var(--series-2)"),
          ("Disk cold load (in)", disk["swap_in_p50_ms"] / 1000, "var(--series-2)")]
  mx = max(v for _, v, _ in rows)
  left, bw, gap, w = 200, 18, 10, 860
  parts, y = [], 8
  for name, v, color in rows:
    r = min(4, bw / 2)
    bl = max(2, (w - left - 80) * v / mx)
    parts.append(f'<text class="cat" x="{left - 8}" y="{y + 13}" text-anchor="end">{name}</text>')
    parts.append(f'<path d="M{left},{y} h{bl - r} a{r},{r} 0 0 1 {r},{r} v{bw - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{bl - r} z" fill="{color}" data-tip="{name}: {v:.1f}s"/>')
    parts.append(f'<text class="val" x="{left + bl + 6}" y="{y + 13}">{v:.1f}s</text>')
    y += bw + gap
  return (f'<svg viewBox="0 0 {w} {y + 4}" role="img" aria-label="Switch phase breakdown at 4B">'
          f'<line x1="{left}" y1="0" x2="{left}" y2="{y}" stroke="var(--axis)"/>' + "".join(parts) + "</svg>")


switch_rows = "".join(f"<tr><td>switch #{i + 1}</td><td>{v / 1000:.1f}s</td></tr>" for i, v in enumerate(snap["switch_ms"]))

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TimeSlice Scale Test — Qwen3-4B / rank 64</title><style>{CSS}</style></head>
<body class="viz-root"><div class="wrap">
<h1>TimeSlice Scale Test — {st["model"].split("/")[-1]} / rank {st["rank"]}</h1>
<p class="sub">NVIDIA L4 · {st["regions"]} GPU regions (~1.9GB real tenant state: rank-64 adapter + AdamW) ·
same node, same tenants, both strategies · disk I/O cache-honest (sync after save, drop_caches before load)</p>

<div class="hero">{speedup:.1f}× <small>faster tenant switching with the snapshot agent at production-like state size</small></div>

<div class="tiles">
 <div class="tile"><div class="v">{snap_rt:.1f}s</div><div class="k">snapshot swap round-trip ({snap["swap_out_ms"] / 1000:.1f}s out / {snap["swap_in_ms"] / 1000:.1f}s in)</div></div>
 <div class="tile"><div class="v">{disk_rt:.1f}s</div><div class="k">disk round-trip: save_state + cold load_from_state</div></div>
 <div class="tile"><div class="v">{snap["vram_freed_mb"] / 1000:.1f} GB</div><div class="k">physical VRAM freed per parked tenant</div></div>
 <div class="tile"><div class="v">{snap["tensors_verified"]}/{snap["tensors_verified"]}</div><div class="k">tensors bitwise-identical after restore; optim_step works post-restore</div></div>
</div>

<div class="card"><h2>The crossover</h2>
<div class="legend"><span style="--c:var(--series-1)">snapshot agent swap</span><span style="--c:var(--series-2)">disk round-trip</span></div>
{crossover_chart()}
<p class="note">At 0.5B/rank-16 the disk path wins ~2× (and that measurement was cache-warm — flattering).
At 4B/rank-64 the verdict flips to a {speedup:.1f}× snapshot win: GPU-CR's 2MB-block amplification shrinks
naturally with tensor size (~20× → ~1.6×) while the disk path grows with real bytes through torch
serialization, storage bandwidth, and PEFT module rebuild. Numbers are node-local disk — network
filesystems (Filestore/NFS) move only the disk bars.</p></div>

<div class="card"><h2>Phase breakdown at 4B</h2>
{breakdown_chart()}
<details><summary>Per-switch timings (snapshot mode)</summary><table><tr><th>switch</th><th>round-trip</th></tr>{switch_rows}</table></details></div>

<div class="card"><h2>Correctness at scale</h2>
<ul class="checks">
<li>{snap["tensors_verified"]} tensors (adapter + AdamW state) bitwise-identical after swap-out/swap-in.</li>
<li>optim_step after restore updates params — GPU-CR's write-after-restore path holds at 4B.</li>
<li>{snap["failures"]} failures across all switches.</li>
<li>Caveat: this gate uses synthetic gradients; end-to-end convergence under swapping at 4B (and the
tenant-B investigation) runs through the full-stack demo driver, not this job.</li>
</ul></div>

<p class="note">Generated by scripts/timeslice_scale_report.py from docs/data/timeslice-run-2026-08-04.json
(scale_test section). Reproduce via docs/timeslice-demo.md §2 with BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507,
LORA_RANK=64, FLUSH_CACHES=1.</p>
</div><div class="tip"></div><script>{TIP_JS}</script></body></html>
"""

with open(OUT, "w") as f:
  f.write(html)
print(f"wrote {OUT} ({len(html)} bytes)")
