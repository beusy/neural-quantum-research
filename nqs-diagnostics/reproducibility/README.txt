DIAGNOSING FAILURES OF NEURAL QUANTUM STATES
Computational research report, May 2026

Read ../report.pdf first. paper.md is the editable
manuscript; main.tex is the standalone LaTeX source. For Overleaf, upload main.tex and the six figures as a new project and select
pdfLaTeX. References are embedded; no BibTeX step is required.

Scope: controlled exact-enumeration spin chains, N=8 and N=10. These are actual
computed results, not a production-scale NQS benchmark or an automatic failure
classifier. The report includes failed interventions and all retained runs.

REPRODUCE
Use Python 3.12 with the versions in environment.json and requirements.txt.
From this directory, with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1:
  python experiments.py
  python witnesses.py
  python symmetry.py
  python optimizers.py
  python sampling.py
  python qgt.py
  python coordinates.py
  python validate.py
  python diagnostic_summary.py
  python make_figures.py
  python write_paper.py
  python build_document.py
The last command requires Pandoc and a LaTeX distribution (pdfLaTeX).
Exact dense diagonalization and retained JSON records make this a deliberately
small-system study. Full regeneration can take several minutes or longer,
depending on hardware. Seeds supplement saved parameters rather than guarantee
identical L-BFGS trajectories across numerical libraries.

CONTENTS
96 primary fits: training_results.json, *_history.json, trained *.npz.
9 symmetry-projected fits: symmetry_results.json, symmetry_w*.npz.
24 Adam/SR comparisons: optimizer_results.json, optimizer_*.npz and histories.
Reference Hamiltonians, eigenvalues, targets: *_reference.npz.
80 frozen sampling repetitions: sampling_results.json; representative
trajectories: samples_*.npz; exact frozen distribution: frozen_sampling_reference.npz.
800 independent Born-sample batches produce 28,800 paired update scores in
qgt_results.json (12 constructions and 3 step radii). These scores are NOT
28,800 independent experiments. Population controls repeat deterministically.
Metric and step normalizations use exact population information for diagnosis;
they are not a deployable stochastic optimizer. See Methods and Appendix C.
coordinate_results.json records invertible diagonal-rescaling controls.
implementation_checks.json and symmetry_gradient_check.json give numerical
independent-Hamiltonian and manually implemented derivative checks.
positive_bound.json and positive_logtable.json concern a basis-dependent
nonnegative-amplitude family, not all real or complex neural states.
figures/ contains six data-generated figures.
../../SHA256SUMS.txt fingerprints the published repository payloads.

NumPy/SciPy supply generic algebra and L-BFGS-B. Neural ansatz derivatives,
Hamiltonians, Metropolis moves, Adam, SR and diagnostic interventions are direct
project implementations. No NetKet, PyTorch or JAX is used. References are in
the report. No supervisory endorsement or external peer review is implied.
