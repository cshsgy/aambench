# Contributing

1. Work on a branch (fork or in-repo), open a PR against `main`. One topic per PR.
2. Derivations go in `doc/chapters/<topic>.tex`; add `\input{chapters/<topic>}` to `doc/main.tex`.
   Tag every claim as *derived* or *measured*.
3. Measured numbers must name: script path, commit sha of this repo (and of snapy/canoe if used),
   norm (e.g. row-0 max |err| / Omega a), state (solid body / W2 / jet), resolution sequence, run length.
4. Scripts are numpy/scipy only unless the chapter says otherwise; keep them runnable in < 5 min on a laptop.
5. Commit messages: no AI attribution trailers. Author = the person whose agent ran it.
6. CI must build the PDF. Do not commit the PDF.
