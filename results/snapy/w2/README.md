# W2 convergence rows (lane T, CPU)

- snapy: chengcli/snapy main `cbf8683009de2d4c211e65e5daeaf1ad1e05c80d`, CPU source build (Release, NETCDF/PNETCDF off), installed into the venv; the package reports version `0.0.0`, the sha is recorded here and in each JSON (`snapy_sha`).
- torch 2.10.0, kintera 2.6.10 wheel, pyharp 2.7.6; Ubuntu 22.04, AMD EPYC 7R13; `OMP_NUM_THREADS=12`, `CUDA_VISIBLE_DEVICES=` (CPU only).
- Command, from outside the source tree, one process per row:
  `python scripts/snapy/w2_convergence.py --mode {off|transport|full} [--closure] --nx {16|32|48|64} --days 62 --every 100 --out results/snapy/w2`
  (the C64 rows ran with `OMP_NUM_THREADS=16` and `--snapy-sha`, so their JSON carries the provenance fields natively)
  (`off` at interp_order 2, `transport` and `full` at interp_order 4; the last step is cut to land on 62.0 d).
- Deck: snapy `tests/test_axial_momentum_flux_form.py` `sw_config` (gnomonic-equiangle, 6 panels, one block per panel, nghost 3, shallow water, weno5 without scale or shock, shallow-roe yz, rk3 cfl 0.5, Coriolis xyz Omega 7.292e-5), Williamson-2 state u0 = 38.6 m/s, phi0 = 2.94e4.
- Norms: AAM `L = sum_interior V r (k2 m2 + k3 m3) + Omega V r^2 cos^2(lat) phi` from the coordinate buffers; `dL/L_end = (L(62 d) - L0) / L0`, `dL/L_max = max over every step of |L - L0| / |L0|`; phi error against the initial steady state over interior cells, `Linf = max|dphi| / phi0`, `L2 = sqrt(sum V dphi^2 / sum V) / phi0`; `dM/M` on `sum V phi`. The sign observation ("no sign change") is on the 100-step series.
- `summary.txt` is the output of `scripts/snapy/w2_summary.py results/snapy/w2`; `thread_gap3_rows.txt` holds the thread rows these reproduce.
- aambench commits: 01ed684 (first nine rows), 72da574 (macro fix), 83ce590 (closure rows), then this review follow-up.
