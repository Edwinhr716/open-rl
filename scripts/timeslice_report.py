"""Aggregate TimeSlice demo metrics into the comparison table.

Usage: python3 scripts/timeslice_report.py <metrics.jsonl> [<more.jsonl> ...]
Groups rows by (mode, event) and prints count/p50/p95 of wall_ms plus VRAM
figures for trainer swaps.
"""

import json
import sys
from collections import defaultdict


def pct(sorted_vals, p):
  if not sorted_vals:
    return float("nan")
  idx = min(len(sorted_vals) - 1, max(0, int(round(p / 100 * len(sorted_vals))) - 1))
  return sorted_vals[idx]


def main(paths):
  rows = []
  for path in paths:
    with open(path) as f:
      for line in f:
        line = line.strip()
        if line:
          try:
            rows.append(json.loads(line))
          except json.JSONDecodeError:
            pass

  groups = defaultdict(list)
  vram = defaultdict(list)
  for r in rows:
    key = (r.get("mode", "-"), r.get("event", "?"))
    if "wall_ms" in r:
      groups[key].append(r["wall_ms"])
    for field in ("vram_freed_mb", "vram_used_mb"):
      if field in r:
        vram[(key, field)].append(r[field])

  print(f"{'mode':<10} {'event':<22} {'n':>4} {'p50_ms':>9} {'p95_ms':>9}")
  print("-" * 58)
  for key in sorted(groups):
    vals = sorted(groups[key])
    print(f"{key[0]:<10} {key[1]:<22} {len(vals):>4} {pct(vals, 50):>9.1f} {pct(vals, 95):>9.1f}")
  for (key, field), vals in sorted(vram.items()):
    svals = sorted(vals)
    print(f"{key[0]:<10} {key[1]:<22}      {field}: p50 {pct(svals, 50):.0f}MB over {len(vals)} swaps")

  summaries = [r for r in rows if r.get("event", "").endswith("summary")]
  if summaries:
    print("\nSummaries:")
    for s in summaries:
      print(" ", json.dumps(s))


if __name__ == "__main__":
  main(sys.argv[1:] if len(sys.argv) > 1 else ["/tmp/timeslice-metrics.jsonl"])
