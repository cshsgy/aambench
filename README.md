# aambench

Reproducible testbed for axial angular momentum (AAM) on the cubed sphere:
panel-seam geometry, the pinned-force seam-row defect, the R_c / D / T budget
identities, and cross-checks against snapy / canoe.

Companion code branch: `aambench` on canoe. This repo holds the derivations
(LaTeX, `doc/`), the standalone numpy experiments (`scripts/`), and recorded
results (`results/`). Every number quoted in `doc/` must point at a script in
`scripts/` and a pinned revision.

## Layout
- `doc/main.tex` – one document, one chapter per topic (`doc/chapters/*.tex`).
- `scripts/geometry/` – six-panel gnomonic geometry and seam-operator tests (numpy only).
- `results/` – committed outputs (text tables) with the commit they came from.
- `.github/workflows/latex.yml` – builds `doc/main.pdf` on every push / PR.

## Run the geometry test
    pip install numpy scipy
    python scripts/geometry/seam_pin.py          # pinned-force seam-row error vs N
    python scripts/geometry/seam_pin_reweight.py # wider seam stencil variant

## Contributing
See CONTRIBUTING.md. Short version: one chapter or script per PR, state the
norm, the state, the grid sequence and the code revision for every number.
