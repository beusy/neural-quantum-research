GEOMETRY-CONDITIONED NEURAL ELECTRONIC STATES

Read ../report.pdf for the complete report.
paper.md and write_paper.py contain the editable manuscript source.

SCIENTIFIC SCOPE
Hydrogen STO-3G electronic Hamiltonians are generated from primitive Gaussian
integrals by chemistry.py. The determinant bases have dimensions 4, 36, 400.
PySCF is used ONLY by validate.py for independent checks. It does not generate
training Hamiltonians, neural gradients, or the main FCI reference matrices.

The neural model outputs signed coefficients in the FULL determinant basis.
It has an exponentially growing readout and is not a scalable autoregressive
NQS implementation. All electronic expectation values are exact configuration
sums; no Monte Carlo results are claimed. Forces are derivatives with respect
to the stated nuclear-spacing coordinates, not full Cartesian force vectors.

REPRODUCTION
Use Python 3.12, NumPy, SciPy, Matplotlib. Exact versions are in environment.json.
Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for all numerical commands.
From the project directory, run:

  python build_data.py
  python rhf_refine.py
  python derivative_refine.py
  python train.py 2 4 6
  python two_dimensional.py
  python response_study.py
  python finalize_scoring.py
  python rescore.py
  python additional_checks.py
  python adaptive.py
  python observables.py
  python gradient_check.py
  python check_2d.py
  python make_figures.py
  python write_paper.py

Optional independent validation:
  python -m pip install pyscf
  python validate.py

PDF build (requires Pandoc and a standard TeX Live installation):
  pandoc paper.md -o geometry_conditioned_electronic_states.pdf --pdf-engine=pdflatex --toc --resource-path=. -V documentclass=article -V papersize=a4

DATA AND RESULTS
H2_data.npz, H4_data.npz, H6_data.npz contain geometry coordinates, complete
Hamiltonian matrices, exact FCI energies and states, fixed-sector gaps, and
scratch RHF values. RHF convergence flags must be honored; 21 H4 points fail
the scratch solver's criterion and are not force references.
H*_derivatives.npz contain final five-point matrix derivatives.
H*_n*_w*_p*_s*.npz contain saved neural parameters, normalized states,
forces, residuals, state tangents, and exact anchor indices.
dimer_data.npz and dimer_n*.npz are the two-coordinate H4 study.
gauge_n*.npz and long_H*.npz contain orbital-twist and longer-budget fits.
*_results.json and the other JSON files contain final scored metrics.

Final one-coordinate metrics EXCLUDE anchor geometries from the ID query set.
All models are retained, including optimizer iteration-limit terminations.
Initial run logs may contain preliminary scores including anchors and centered
matrix derivatives. Final JSON files and rescored NPZ curves are authoritative.
finalize_scoring.py and rescore.py apply the final conventions.

CAVEATS
The original reference-grid code initially used a damped SCF baseline; the
final RHF implementation uses DIIS, with explicit convergence statuses.
Baselines.json retains a scratch RHF row for diagnostics, but manuscript
Tables 2/5 do not use its full force curve. Converged RHF points are shown only
as an illustrative energy reference.

The validation script uses a zero memory-reporting fallback only if its
container exposes a /proc mount incompatible with the current process ID.
This affects library memory-budget reporting, not integrals or wavefunctions.

Third-party articles used as references are NOT included. No FCI energy,
coefficient, force, or observable labels are training targets. Classical
coefficient-spline baselines DO use FCI anchor states and are explicitly
information-rich controls. Candidate collocation uses precomputed Hamiltonians
and is not a claim of reduced ab-initio labeling cost.
