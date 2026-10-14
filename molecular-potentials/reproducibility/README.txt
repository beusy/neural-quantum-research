LEARNING CONSERVATIVE MOLECULAR POTENTIALS FROM SCRATCH

Read ../report.pdf for the research report. paper.md is its editable source.

REPRODUCTION
Python 3.12, NumPy, SciPy, Matplotlib; see environment.json for recorded versions.
From this directory:
  python download_data.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python extensions.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python diagnostics.py
  python make_figures.py
  python write_paper.py
  pandoc paper.md -o paper.pdf --pdf-engine=pdflatex --toc --resource-path=.

The dataset downloader retrieves approximately 67 MB from the public rMD17
repository. The dataset itself is not bundled. Its exact SHA-256 is verified.

CONTENTS
experiments.py: H2 models; bound dynamics; RBF uncertainty ensemble;
               scalar/vector features, synthetic H3, rMD17 ethanol.
extensions.py: energy/force derivative-kernel baseline and a fully trainable
               19-parameter scalar/vector network on bounded synthetic H3.
diagnostics.py: chronological split, force-feature rank, mixed derivative
                checks, controlled active selection.
make_figures.py: additional acquisition/MD figure.
results.json, extensions.json, diagnostics.json: measured metrics.
splits.npz, chronological_splits.npz: exact archive indices.
network_checks.json: scalar/vector neural symmetry checks.
run.log: main experiment execution.
extensions.log: initial kernel search and initial H3 run (historical).
kernel_refinement.log: expanded kernel search, supersedes initial kernel rows.
h3_controlled.log: matched initialization H3 run, supersedes initial H3 rows.

extensions.json contains the FINAL kernel search and matched-init H3 results.
A fresh run of extensions.py uses the FINAL settings directly. The wall-time
metadata in the preserved JSON records both the initial and later executions;
these timing fields will differ on a fresh run. Neural termination at the
iteration budget is reported as unsuccessful convergence, not hidden.

SCIENTIFIC BOUNDARIES
Hydrogen references are analytic synthetic functions, not DFT hydrogen data.
The ethanol splits are custom; no official benchmark reproduction is claimed.
The fixed-order inverse-distance kernel is not FCHL19 or sGDML.
The fixed scalar/vector ethanol feature model has a linear readout; the
end-to-end trainable scalar/vector neural model was run on synthetic H3 only.
No molecular MD, aspirin/paracetamol training, or new quantum chemistry was run.
Original research directions are proposals, not claims of literature priority.
