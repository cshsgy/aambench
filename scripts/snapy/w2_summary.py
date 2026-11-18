#!/usr/bin/env python3
"""Summarise results/snapy/w2/w2_<mode>_C<nx>.json from w2_convergence.py: phi errors, observed
orders between successive resolutions, and the AAM drift of off over transport and full,
by end value and by per-step maximum.

  python w2_summary.py results/snapy/w2 > results/snapy/w2/summary.txt
"""
import glob
import json
import math
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
runs = {}
for p in sorted(glob.glob(os.path.join(d, "w2_*_C*.json"))):
    r = json.load(open(p))
    runs[(r["mode"], r["nx"])] = r
modes = [m for m in ("off", "transport", "full", "full+closure") if any(k[0] == m for k in runs)]
nxs = sorted({k[1] for k in runs})
r0 = next(iter(runs.values()))
print("Williamson 2 at the Earth's rate, %g days, snapy %s, torch %s, device %s"
      % (r0["days"], r0["snapy"], r0["torch"], r0["device"]))
print("%-10s %4s %4s %7s %10s %10s %12s %12s %9s %7s" % ("mode", "C", "io", "cycles", "phi_Linf", "phi_L2",
                                                       "dL/L_end", "dL/L_max", "dM/M", "wall_s"))
for m in modes:
    for nx in nxs:
        r = runs.get((m, nx))
        if r:
            print("%-10s %4d %4s %7d %10.3e %10.3e %+12.3e %12.3e %9.1e %7.0f"
                  % (m, nx, r["interp_order"], r["cycles"], r["phi_linf"], r["phi_l2"], r["dL_over_L_end"],
                     r["dL_over_L_max"], r["dM_over_M"], r["wall_s"]))


def order(a, b, ha, hb):
    return math.log(a / b) / math.log(hb / ha)  # h = 1 / C


print("\nobserved order of the phi error between successive resolutions (Linf / L2)")
for m in modes:
    row = []
    for i in range(len(nxs) - 1):
        a, b = runs.get((m, nxs[i])), runs.get((m, nxs[i + 1]))
        if a and b:
            row.append("C%d->C%d %.2f / %.2f" % (nxs[i], nxs[i + 1], order(a["phi_linf"], b["phi_linf"], nxs[i], nxs[i + 1]),
                                                 order(a["phi_l2"], b["phi_l2"], nxs[i], nxs[i + 1])))
    print("%-10s %s" % (m, "   ".join(row)))

print("\nphi error of the mode over off (Linf), and AAM drift of off over the mode, by end value / by per-step max")
for m in modes:
    if m == "off":
        continue
    row = []
    for nx in nxs:
        a, b = runs.get(("off", nx)), runs.get((m, nx))
        if a and b:
            if "closure" in m:  # the closure drift is round-off: no ratio
                row.append("C%d err %.2f, AAM at round-off" % (nx, b["phi_linf"] / a["phi_linf"]))
                continue
            r1, r2 = abs(a["dL_over_L_end"]) / abs(b["dL_over_L_end"]), a["dL_over_L_max"] / b["dL_over_L_max"]
            fmt = "%.1fx" if max(r1, r2) < 10 else "%.0fx"
            row.append(("C%d err %.2f, AAM " + fmt + " / " + fmt) % (nx, b["phi_linf"] / a["phi_linf"], r1, r2))
    print("%-10s %s" % (m, "   ".join(row)))
