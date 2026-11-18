#!/usr/bin/env python3
"""Lane T: Williamson 2 (steady zonal flow, Earth's rate) on the cubed sphere to 62 days,
C16 / C32 / C48, with conserve-axial-momentum off, transport and full (the latter two with
geometry/cells/interp_order: 4), on the pinned snapy main. Reproduces the thread's claim
(Jiheng, 11-14, gap 3 on xiz/full-on-357 1be61abd): full's phi error below off at every
resolution with observed order ~1.8 / 2.1, and full's absolute-AAM drift 132x / 271x / 414x
below off by end value (80x / 146x / 215x by per-step max).

Deck: snapy's tests/test_axial_momentum_flux_form.py sw_config (shallow water, weno5 with
scale and shock off, shallow-roe yz, rk3 cfl 0.5, Omega 7.292e-5, u0 38.6 m/s, phi0 2.94e4),
one block per panel, nx2 = nx3 = C, nghost 3. The AAM is read from the coordinate buffers
(relative part V r k.m plus planetary omega V r^2 cos^2(lat) phi), interior cells only, as the
test does. Error: phi against the initial steady state, interior cells, Linf and L2
(volume-weighted rms), both relative to phi0.

  python w2_convergence.py --mode full --nx 16 --days 62 --out results/snapy/w2
Writes <out>/w2_<mode>_C<nx>.json with the per-step series and the summary line.
"""
import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time

import torch
import yaml

SW_OMEGA, SW_U0, SW_PHI0 = 7.292e-5, 38.6, 2.94e4  # Williamson et al. (1992) test 2
A_EARTH = 6371000.0
DAY = 86400.0


def sw_config(mode, nx, omega=SW_OMEGA, interp_order=None):
    edge = {"x2min_pi": -0.25, "x2max_pi": 0.25, "x3min_pi": -0.25, "x3max_pi": 0.25}
    cells = {"nx1": 1, "nx2": nx, "nx3": nx, "nghost": 3}
    if interp_order is not None:
        cells["interp_order"] = interp_order
    return {
        "geometry": {"type": "gnomonic-equiangle",
                     "bounds": dict(x1min=A_EARTH, x1max=A_EARTH + 1.0, **edge),
                     "cells": cells},
        "distribute": {"layout": "cubed-sphere", "nb2": 1, "nb3": 1, "blocks_per_process": 6, "verbose": False},
        "dynamics": {"conserve-axial-momentum": mode,
                     "equation-of-state": {"type": "shallow-water"},
                     "reconstruct": {"vertical": {"type": "weno5", "scale": False, "shock": False},
                                     "horizontal": {"type": "weno5", "scale": False, "shock": False}},
                     "riemann-solver": {"type": "shallow-roe", "dir": "yz"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": "custom", "x2-outer": "custom",
                                            "x3-inner": "custom", "x3-outer": "custom"}},
        "integration": {"type": "rk3", "cfl": 0.5, "nlim": 10 ** 9, "tlim": 1.0e12, "ncycle_out": 0},
        "forcing": {"coriolis": {"type": "xyz", "omega1": omega}},
    }


def build(cfg, device):
    from snapy import Mesh, MeshOptions

    nx = cfg["geometry"]["cells"]["nx2"]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        options = MeshOptions.from_yaml(tmp)
        options.blocks_per_process(6)
        options.set_local_horizontal_cells(nx, nx)
        mesh = Mesh(options)
    finally:
        os.unlink(tmp)
    blocks = list(mesh.blocks)
    for block in blocks:
        block.to(torch.device(device))
    return mesh, blocks


def interior(block):
    return tuple(block.part((0, 0, 0), False)[1:])


def sw_state(blocks, omega=SW_OMEGA, u0=SW_U0):
    from snapy import kIDN, kIV2, kIV3

    mesh_vars = []
    for block in blocks:
        coord = block.module("coord")
        w = dict(block.named_buffers())["hydro.D"].clone().zero_()
        lat = coord.buffer("latitude").to(w)
        a = coord.buffer("x1v").to(w)
        k = coord.buffer("aam_k_c").to(w)
        w[kIDN] = SW_PHI0 - (a * omega * u0 + 0.5 * u0 ** 2) * torch.sin(lat) ** 2
        w[kIV2] = u0 * k[1] + 0.0 * a
        w[kIV3] = u0 * k[2] + 0.0 * a
        mesh_vars.append({"hydro_w": w})
    return mesh_vars


def aam(blocks, mesh_vars, omega):
    from snapy import kIDN, kIV2, kIV3

    total = 0.0
    for ib, block in enumerate(blocks):
        coord = block.module("coord")
        u = mesh_vars[ib]["hydro_u"].double()
        k = coord.buffer("aam_k_c").double()
        r = coord.buffer("x1v").double()
        vol = coord.cell_volume().double()
        l = vol * r * (k[1] * u[kIV2] + k[2] * u[kIV3])
        l = l + omega * vol * r * r * coord.buffer("aam_kk_c").double() * u[kIDN]
        total += l[interior(block)].sum().item()
    return total


def mass(blocks, mesh_vars):
    from snapy import kIDN

    return sum((b.module("coord").cell_volume().double() * mesh_vars[ib]["hydro_u"][kIDN].double())
               [interior(b)].sum().item() for ib, b in enumerate(blocks))


def phi_error(blocks, mesh_vars, phi0):
    from snapy import kIDN

    linf, l2num, l2den = 0.0, 0.0, 0.0
    for ib, b in enumerate(blocks):
        d = (mesh_vars[ib]["hydro_u"][kIDN].double()[interior(b)] - phi0[ib])
        vol = b.module("coord").cell_volume().double()[interior(b)]
        linf = max(linf, d.abs().max().item())
        l2num += (vol * d * d).sum().item()
        l2den += vol.sum().item()
    return linf / SW_PHI0, math.sqrt(l2num / l2den) / SW_PHI0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["off", "transport", "full"])
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--days", type=float, default=62.0)
    ap.add_argument("--interp-order", type=int, default=None,
                    help="default: 4 for transport and full, unset (2) for off")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=".")
    ap.add_argument("--every", type=int, default=50, help="store the series every N steps")
    ap.add_argument("--closure", action="store_true", help="dynamics/conserve-axial-momentum-closure: true (full only, #365)")
    args = ap.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    io = args.interp_order if args.interp_order is not None else (None if args.mode == "off" else 4)
    cfg = sw_config(args.mode, args.nx, interp_order=io)
    if args.closure:
        cfg["dynamics"]["conserve-axial-momentum-closure"] = True
    mesh, blocks = build(cfg, args.device)
    from snapy import kIDN

    mesh_vars, _ = mesh.initialize(sw_state(blocks))
    intg = blocks[0].module("intg")
    phi0 = [mesh_vars[ib]["hydro_u"][kIDN].double()[interior(b)].clone() for ib, b in enumerate(blocks)]
    l0, m0 = aam(blocks, mesh_vars, SW_OMEGA), mass(blocks, mesh_vars)
    t, cycle, drift_max, series = 0.0, 0, 0.0, []
    t0 = time.time()
    tend = args.days * DAY
    while t < tend - 1.0e-9:
        cycle += 1
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        if t + dt > tend:
            dt = tend - t  # land exactly on the end time
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
        if mesh.check_redo(mesh_vars) != 0:
            raise RuntimeError("step rejected at cycle %d" % cycle)
        t += dt
        l = aam(blocks, mesh_vars, SW_OMEGA)
        d = (l - l0) / l0
        if not math.isfinite(d):
            raise RuntimeError("non-finite AAM at cycle %d" % cycle)
        drift_max = max(drift_max, abs(d))
        if cycle % args.every == 0 or t >= tend - 1.0e-9:
            linf, l2 = phi_error(blocks, mesh_vars, phi0)
            series.append({"cycle": cycle, "t_days": t / DAY, "dt": dt, "dL_over_L": d,
                           "phi_linf": linf, "phi_l2": l2, "dM_over_M": (mass(blocks, mesh_vars) - m0) / m0})
            print("cycle %6d  t %7.3f d  dt %8.2f  dL/L %+.3e  |dL/L|max %.3e  Linf %.3e  L2 %.3e  (%.0f s)"
                  % (cycle, t / DAY, dt, d, drift_max, linf, l2, time.time() - t0), flush=True)
    linf, l2 = phi_error(blocks, mesh_vars, phi0)
    import snapy
    out = {"mode": args.mode + ("+closure" if args.closure else ""), "nx": args.nx, "interp_order": io, "days": args.days, "device": args.device,
           "snapy": getattr(snapy, "__version__", "?"), "torch": torch.__version__,
           "cycles": cycle, "dt_first": series[0]["dt"] if series else None, "L0": l0,
           "dL_over_L_end": (aam(blocks, mesh_vars, SW_OMEGA) - l0) / l0, "dL_over_L_max": drift_max,
           "phi_linf": linf, "phi_l2": l2, "dM_over_M": (mass(blocks, mesh_vars) - m0) / m0,
           "wall_s": time.time() - t0, "series": series}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "w2_%s_C%d.json" % (out["mode"], args.nx))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("SUMMARY %s C%d io%s %g d: cycles %d, dL/L end %+.3e, max %.3e, phi Linf %.3e, L2 %.3e, dM/M %.1e, %.0f s -> %s"
          % (out["mode"], args.nx, io, args.days, cycle, out["dL_over_L_end"], drift_max, linf, l2, out["dM_over_M"],
             out["wall_s"], path), flush=True)
    del mesh, blocks, mesh_vars
    gc.collect()


if __name__ == "__main__":
    main()
