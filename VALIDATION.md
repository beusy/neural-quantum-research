# Validation and packaging notes

## Publication checks

- The two supplied source-archive checksum lists were verified before editing: no mismatches.
- All three selected report PDFs retain exactly the supplied bytes and name Don van Beusichem in their title pages and PDF metadata.
- All Python source files parse, and repository navigation links were checked.
- A scan for common credential patterns found no matches; this is not a comprehensive security audit.
- Independent small-system Hamiltonian constructions, neural objective derivatives, normalized tangents, and the QGT covariance identity were checked using the supplied diagnostic validation script.
- The electronic-state mixed parameter-gradient check passed after correcting its output path.
- Additional molecular-model derivative checks and structured-data checks are recorded in `publication-checks.json`.

Validation ran on copies of the experiments, leaving the supplied scientific result files unchanged. The checking environment may differ from the archived environment; its NumPy and SciPy versions are recorded in `publication-checks.json`.

Full training, all saved metrics, optional PySCF comparisons, manuscript bibliography, and the external rMD17 download were not independently reproduced or verified during publication. The retained result files describe the supplied runs, not new runs performed during this packaging step.

## Packaging changes

The three archives were extracted into descriptive study folders. Internal numerical file locations were retained. The corrected author-supplied PDFs replace older anonymous/generic-author report copies; the duplicate diagnostics PDF was not published twice. Manuscript and generator author fields now name Don van Beusichem. Study dates were corrected to January, March, and May 2026 as confirmed by the author. Technical environment versions, random seeds, experiment results, and scientific references were retained.

Two electronic-state validation scripts now write next to their source files rather than to a hard-coded `idea1/` directory. Numerical routines were not changed.

Obsolete PDF build logs and old checksum manifests were omitted. Scientific execution logs, figures, saved models, and result records are retained. A fresh repository-wide checksum manifest replaces the original manifests, which would no longer describe the edited package.

Manuscript sources are working versions supplied with the archives; they need not reproduce the corrected PDF editions word for word or with identical formatting. The author-supplied PDFs are the primary reports.
