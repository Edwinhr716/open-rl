"""Generate the tenant-ceiling campaign report (tabbed: Results / Setup).

Three arms at Qwen3-4B / rank-48 on one 2xL4 node: stock Open-RL (all
tenants resident) to its OOM wall, Open-RL + disk eviction at 30 tenants,
and the Snapshot Agent at 28 tenants on a RAM-only (tmpfs) store.

Usage: python3 scripts/timeslice_tenant_ceiling_report.py
"""

import json

d = json.load(open("docs/data/tenant-ceiling-2026-08-06.json"))
OUT = "docs/timeslice-tenant-ceiling-report.html"
S1, S2, S3 = "var(--series-1)", "var(--series-2)", "var(--series-3)"

CSS = """
.viz-root { color-scheme: light;
  --page:#f9f9f7; --surface-1:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --ok:#006300; --bad:#d03b3b; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
    --page:#0d0d0d; --surface-1:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --ok:#0ca30c; --bad:#e66767; } }
.viz-root { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page); color: var(--ink); margin: 0; padding: 24px; }
.wrap { max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 0 0 2px; }
.sub { color: var(--ink-2); font-size: 13px; margin: 0 0 12px; }
.hero { font-size: 40px; font-weight: 700; margin: 12px 0 2px; }
.hero small { font-size: 15px; font-weight: 500; color: var(--ink-2); }
.tabs { display: flex; gap: 4px; margin: 16px 0 12px; border-bottom: 1px solid var(--grid); }
.tab { padding: 8px 18px; font-size: 13.5px; cursor: pointer; border: 1px solid transparent;
  border-bottom: none; border-radius: 8px 8px 0 0; color: var(--ink-2); }
.tab.active { background: var(--surface-1); border-color: var(--grid); color: var(--ink); font-weight: 600; }
.pane { display: none; } .pane.active { display: block; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 12px; margin: 14px 0; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .v { font-size: 24px; font-weight: 650; } .tile .k { color: var(--ink-2); font-size: 12px; margin-top: 2px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin: 10px 0; }
.legend { display: flex; gap: 16px; font-size: 12px; color: var(--ink-2); margin: 4px 0 8px; flex-wrap: wrap; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 5px; vertical-align: -1px; background: var(--c); }
svg text { font-family: inherit; font-size: 11px; fill: var(--muted); }
svg .val { fill: var(--ink-2); font-variant-numeric: tabular-nums; }
svg .cat { fill: var(--ink-2); }
svg .bad { fill: var(--bad); font-weight: 700; }
table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; }
.note { color: var(--muted); font-size: 11.5px; margin-top: 6px; }
.ok { color: var(--ok); font-weight: 600; }
ol.set, ul.set { font-size: 13px; color: var(--ink-2); padding-left: 18px; } ol.set li, ul.set li { margin-bottom: 6px; }
code { background: var(--grid); border-radius: 4px; padding: 0 4px; font-size: 12px; }
.tip { position: fixed; pointer-events: none; background: var(--ink); color: var(--page);
  padding: 4px 8px; border-radius: 6px; font-size: 12px; opacity: 0; transition: opacity .08s; z-index: 9; }
"""

JS = """
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab,.pane').forEach(x => x.classList.remove('active'));
  t.classList.add('active'); document.getElementById(t.dataset.pane).classList.add('active');
}));
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


def chart_ceiling():
  bars = [("Stock Open-RL", d["stock"]["wall_tenants"], S2, "CUDA OOM creating tenant 11"),
          ("Open-RL + disk eviction", d["disk"]["tenants"], S3, f"30/30 created, {d['disk']['rt_s']:.1f}s per switch"),
          ("Snapshot Agent (RAM store)", d["snap"]["tenants"], S1, f"28 tenants, {d['snap']['switch_p50_ms']/1000:.1f}s switches, all state in RAM")]
  mx = 32
  w, h, pad_b, pad_t, left = 920, 250, 40, 14, 40
  gw = (w - left) / 3
  parts = [f'<line x1="{left}" y1="{h - pad_b}" x2="{w - 10}" y2="{h - pad_b}" stroke="var(--axis)"/>']
  for gl in (10, 20, 30):
    gy = h - pad_b - (h - pad_b - pad_t) * gl / mx
    parts.append(f'<line x1="{left}" y1="{gy}" x2="{w - 10}" y2="{gy}" stroke="var(--grid)"/>')
    parts.append(f'<text x="{left - 4}" y="{gy + 4}" text-anchor="end">{gl}</text>')
  for i, (name, v, color, tipv) in enumerate(bars):
    x = left + 60 + i * gw
    bh = (h - pad_b - pad_t) * v / mx
    parts.append(rbar_v(x, h - pad_b - bh, 110, bh, color, tipv))
    parts.append(f'<text class="val" x="{x + 55}" y="{h - pad_b - bh - 6}" text-anchor="middle" style="font-size:14px;font-weight:600">{v}</text>')
    if i == 0:
      parts.append(f'<text class="bad" x="{x + 55}" y="{h - pad_b - bh - 22}" text-anchor="middle">OOM ✗</text>')
    parts.append(f'<text class="cat" x="{x + 55}" y="{h - pad_b + 15}" text-anchor="middle">{name}</text>')
  return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Tenant ceiling by strategy">' + "".join(parts) + "</svg>"


def chart_switch():
  bars = [("Stock (until the wall)", 0.001, S2, "0ms — resident set_adapter flag flip; works only below 11 tenants"),
          ("Disk eviction round-trip", d["disk"]["rt_s"], S3, f"{d['disk']['out_p50_ms']/1000:.1f}s park + {d['disk']['in_p50_ms']/1000:.1f}s reload, cache-honest"),
          ("Snapshot Agent round-trip", d["snap"]["switch_p50_ms"] / 1000, S1, f"p50 over {d['snap']['switch_n']} live switches ({d['snap']['out_p50_ms']/1000:.1f}s out / {d['snap']['in_p50_ms']/1000:.1f}s in), RAM store")]
  mx = max(v for _, v, _, _ in bars)
  left, bw, gap, w = 250, 20, 12, 920
  parts, y = [], 8
  for name, v, color, tipv in bars:
    bl = max(2.5, (w - left - 110) * v / mx)
    r = min(4, bw / 2)
    parts.append(f'<text class="cat" x="{left - 8}" y="{y + 14}" text-anchor="end">{name}</text>')
    parts.append(f'<path d="M{left},{y} h{bl - r} a{r},{r} 0 0 1 {r},{r} v{bw - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{bl - r} z" fill="{color}" data-tip="{tipv}"/>')
    lbl = "~0" if v < 0.01 else f"{v:.1f}s"
    parts.append(f'<text class="val" x="{left + bl + 6}" y="{y + 14}">{lbl}</text>')
    y += bw + gap
  return (f'<svg viewBox="0 0 {w} {y + 4}" role="img" aria-label="Tenant switch cost by strategy">'
          f'<line x1="{left}" y1="0" x2="{left}" y2="{y}" stroke="var(--axis)"/>' + "".join(parts) + "</svg>")


results = f"""
<div class="tiles">
 <div class="tile"><div class="v">2.8×</div><div class="k">more tenants than stock Open-RL (28 vs 10) with all parked state in RAM</div></div>
 <div class="tile"><div class="v">{d["snap"]["switch_p50_ms"]/1000:.1f}s</div><div class="k">snapshot switch p50 across {d["snap"]["switch_n"]} live switches — {d["disk"]["rt_s"]/(d["snap"]["switch_p50_ms"]/1000):.1f}× faster than disk eviction</div></div>
 <div class="tile"><div class="v">{d["snap"]["vram_freed_p50_mb"]/1000:.1f} GB</div><div class="k">VRAM freed per parked tenant (p50), VRAM flat for the whole run</div></div>
 <div class="tile"><div class="v">{d["snap"]["retention"]}</div><div class="k">tenants reproduced their trained outputs in the final retention sweep</div></div>
</div>

<div class="card"><h2>Tenant ceiling — Qwen3-4B, rank-48, one L4 per role</h2>
{chart_ceiling()}
<p class="note">Stock Open-RL keeps every tenant's adapter + AdamW state resident and hits
<code>torch.OutOfMemoryError</code> creating tenant 11 (22.02/22.03 GiB in use). Both eviction strategies
sail past it; the snapshot arm stopped at 28 when the run budget ended (store headroom remained after a
mid-run limit bump). Context points measured en route: at rank-64 the stock wall is 8; with GPU-CR's
allocator taxes applied to a resident design, it is 2–3.</p></div>

<div class="card"><h2>What a tenant switch costs</h2>
<div class="legend"><span style="--c:{S2}">stock (resident)</span><span style="--c:{S3}">disk eviction</span><span style="--c:{S1}">snapshot agent</span></div>
{chart_switch()}
<p class="note">Stock switching is a flag flip — but only exists below the wall. Disk eviction
(save_state + cold load_from_state, sync + drop_caches) pays serialization, storage bandwidth, and PEFT
rebuild. The snapshot agent moves raw GPU state through hugetlbfs and a tmpfs store: no serialization, no
storage, no rebuild — {d["disk"]["rt_s"]/(d["snap"]["switch_p50_ms"]/1000):.1f}× faster at the same tenant count, and unlike the disk numbers these are
from live RL traffic (126 switches during real training), not a synthetic cycle.</p></div>

<div class="card"><h2>Convergence and retention under swapping</h2>
<table><tr><th>check</th><th>result</th></tr>
<tr><td>Driver pairs completed (each with temp-0 determinism tripwires)</td><td class="ok">14/14 PASSED</td></tr>
<tr><td>Target-token rate by round (aggregate, both tenants)</td><td>r0 {d["snap"]["conv_pct_by_round"]["0"]:.0f}% → r1 {d["snap"]["conv_pct_by_round"]["1"]:.0f}% → r2 {d["snap"]["conv_pct_by_round"]["2"]:.0f}%</td></tr>
<tr><td>Final retention sweep: every tenant re-sampled after the full campaign</td><td class="ok">{d["snap"]["retention"]} exact trained-output reproduction</td></tr></table>
<p class="note">Every tenant trains toward its own constant target (420 or 777) for 3 rounds; by round 2 each
emits its trained pattern (A-tenants converge to a 420-based oscillation, B-tenants to pure 777s — the r2
aggregate of 75% reflects that mix at the token level; the retention check compares exact sequences). The
sweep reloads each of the 28 tenants hours after parking: zero drift, zero cross-tenant contamination. An
initial sweep pass produced 13 transient gateway 400s under back-to-back reload load; all 13 reproduced
exact outputs on a paced retry.</p></div>

<div class="card"><h2>All numbers</h2>
<table>
<tr><th>metric</th><th>stock</th><th>disk eviction</th><th>snapshot agent</th></tr>
<tr><td>Tenant ceiling (measured)</td><td>10 (OOM)</td><td>30 (run target)</td><td>28 (run budget)</td></tr>
<tr><td>Switch round-trip p50</td><td>~0 below wall</td><td>{d["disk"]["rt_s"]:.1f}s</td><td>{d["snap"]["switch_p50_ms"]/1000:.2f}s</td></tr>
<tr><td>Park / restore p50</td><td>—</td><td>{d["disk"]["out_p50_ms"]/1000:.1f}s / {d["disk"]["in_p50_ms"]/1000:.1f}s</td><td>{d["snap"]["out_p50_ms"]/1000:.1f}s / {d["snap"]["in_p50_ms"]/1000:.2f}s</td></tr>
<tr><td>VRAM freed per parked tenant</td><td>0</td><td>real footprint</td><td>{d["snap"]["vram_freed_p50_mb"]/1000:.1f} GB p50</td></tr>
<tr><td>Parked-state location</td><td>— (crashes)</td><td>boot disk (~1GB/tenant)</td><td>tmpfs RAM ({d["snap"]["store_gb"]}G total)</td></tr>
<tr><td>Correctness</td><td>n/a</td><td>0 failures, reload byte-exact</td><td>14/14 runs, tripwires green, {d["snap"]["retention"]} retention</td></tr>
</table></div>
"""

setup = f"""
<div class="card"><h2>Why rank 48 (and the bug that decided it)</h2>
<ul class="set">
<li>The unrounded-dump GPU-CR build crashes on any single tensor larger than one 2MB VMM block
(<code>cudaMemcpyAsync</code> → <code>invalid argument</code> at nv.cpp:133). At rank 64 on Qwen3-4B, fp32
AdamW moments for gate/up projections are 9728×64×4B = 2.38MB — the first such tensors ever swapped. Rank 48
keeps every tensor under 2MB (max 1.87MB) and passes cleanly. Tracked as an open unrounded-branch bug with
the rank-64 crash log as reproducer; all three arms were measured at rank 48 for apples-to-apples.</li>
</ul></div>

<div class="card"><h2>Right-sizing the node (the SHM_SIZE_GB=8 build)</h2>
<ul class="set">
<li>GPU-CR reserves a compile-time dump buffer per CUDA process. The default 25GiB (+2×1GiB staging) forced a
60GiB hugepage carve-out per node, leaving only ~27Gi of RAM. The <code>.so</code> was rebuilt with the
supported cmake flag <code>-DSHM_SIZE_GB=8</code> (image <code>gpucr-so:unrounded-shm8</code>) — no source
changes.</li>
<li>Node pool recreated with <code>hugepage_size2m: 12288</code> (24Gi: 2 GPU processes × 10Gi + agent
slack): <b>allocatable RAM went from ~27Gi to ~63Gi</b>, which is what makes a RAM-only snapshot store for
~30 tenants possible at all. Pod hugepage requests dropped from 28Gi to 11Gi each.</li>
</ul>
<table><tr><th>96GB node budget</th><th>before</th><th>campaign</th></tr>
<tr><td>Hugepage reservation</td><td>60Gi</td><td>24Gi</td></tr>
<tr><td>Allocatable RAM</td><td>~27Gi</td><td>~63Gi</td></tr>
<tr><td>Snapshot store (tmpfs, agent-managed)</td><td>≤12Gi</td><td>41G used at 28 tenants (~1.5GB/tenant)</td></tr></table></div>

<div class="card"><h2>Stack configuration</h2>
<ul class="set">
<li><b>Images:</b> workload <code>timeslice-sampler:v11-shm8</code> (vLLM 0.22 base, unrounded shm8
<code>.so</code>, swap-in-first tenant manager); agent <code>snapshot-agent:hugetlbfs-v6</code>
(hole-faithful restore copies, stale-artifact GC, 120s op timeouts, FAULTED auto-recovery), burstable
memory 4Gi request / 46Gi limit for the tmpfs store, <code>SNAPSHOT_DIR=/dev/shm/gcr-snapshots</code>.</li>
<li><b>Topology:</b> single g2-standard-24 (2×L4): trainer on one GPU, vLLM sampler on the other, both
pinned by nodeSelector (hostPath state is node-local); gateway + redis on CPU nodes.</li>
<li><b>Arms:</b> stock = <code>TIMESLICE_*=0</code> with all GPU-CR env removed (fair conditions);
disk = the save_state/load_from_state harness, <code>FLUSH_CACHES=1</code>, 30 tenants;
snapshot = 15 driver pairs (<code>ROUNDS=3</code>, <code>LORA_RANK=48</code>, lr 1e-3 + grad clip),
tenants accumulate via <code>create_model</code> per pair — no code changes, env/config only.</li>
<li><b>Run hygiene (mandatory):</b> clean-slate wipe between arms (hugetlbfs dump files pin reservations
5–15 min past process death; rapid cycling exhausts the 24Gi pool and fakes init failures).</li>
</ul></div>
"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TimeSlice — Tenant Ceiling: Stock vs Disk vs Snapshot Agent</title><style>{CSS}</style></head>
<body class="viz-root"><div class="wrap">
<h1>Tenant Ceiling at Qwen3-4B — Stock Open-RL vs Disk Eviction vs Snapshot Agent</h1>
<p class="sub">2026-08-06 · Qwen3-4B-Instruct-2507 · rank-48 LoRA · one 2×L4 node · live RL traffic through the
full Open-RL stack · no Open-RL code changes (env/config only)</p>
<div class="hero">28 tenants in RAM. <small>Stock crashes at 10; disk gets there 5.5× slower.</small></div>
<div class="tabs">
 <div class="tab active" data-pane="results">Results</div>
 <div class="tab" data-pane="setup">Setup</div>
</div>
<div class="pane active" id="results">{results}</div>
<div class="pane" id="setup">{setup}</div>
<p class="note">Generated by scripts/timeslice_tenant_ceiling_report.py from
docs/data/tenant-ceiling-2026-08-06.json. Companion pages: timeslice-full-report.html (all prior runs),
timeslice-demo.md (reproduction guide).</p>
</div><div class="tip"></div><script>{JS}</script></body></html>
"""

open(OUT, "w").write(html)
print("wrote", OUT, len(html), "bytes")
