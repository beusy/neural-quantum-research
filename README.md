# Neural Quantum States and Molecular Potentials

**Don van Beusichem**

Three computational studies of learned molecular energies and neural representations of quantum states, with reports, implementations, saved experiments, and reproducibility materials.

## Studies

| Study | Focus | Report | Code and results |
|---|---|---|---|
| [Conservative molecular potentials](molecular-potentials/) | Derivative supervision, symmetry-aware descriptors, and extrapolation | [PDF](molecular-potentials/report.pdf) | [Materials](molecular-potentials/reproducibility/) |
| [Geometry-conditioned electronic states](geometry-conditioned-states/) | Variational states, response forces, and orbital-chart dependence | [PDF](geometry-conditioned-states/report.pdf) | [Materials](geometry-conditioned-states/reproducibility/) |
| [Neural quantum-state diagnostics](nqs-diagnostics/) | Representation, optimization, sampling, and quantum geometry | [PDF](nqs-diagnostics/report.pdf) | [Materials](nqs-diagnostics/reproducibility/) |

## Scientific scope

The molecular-potentials study combines analytic hydrogen reference surfaces with a small rMD17 ethanol experiment. Synthetic hydrogen data are not electronic-structure calculations. The ethanol experiments use custom splits and do not claim state-of-the-art benchmark performance.

The electronic-state study enumerates finite determinant spaces for hydrogen systems in STO-3G. Exact references are exact within that finite basis; the architecture is not presented as a scalable many-electron solver. Reported forces refer to the stated geometry coordinates.

The diagnostic study uses small spin systems with exact enumeration to isolate failure mechanisms. It is a set of controlled experiments, not an automatic diagnostic classifier or a claim of scalable optimization advantage.

## Reproducibility

Each study contains its own requirements, recorded environment, execution instructions, and retained results. Begin with its README before launching experiments. Full reruns may overwrite stored result files; use a separate working copy if preserving the supplied snapshot.

The rMD17 ethanol dataset is downloaded by the molecular study's script and checked against a recorded SHA-256 hash. It is not bundled. Optional PySCF validation is described in the electronic-state instructions.

[Validation and packaging notes](VALIDATION.md) distinguish checks performed for this publication from the original experiment records. [SHA256SUMS.txt](SHA256SUMS.txt) records the published file hashes.

## Authorship and acknowledgment

**Author: Don van Beusichem.** I thank **Filippo Vicentini** for informal guidance and helpful discussions. This acknowledgment does not imply a formal supervision arrangement, coauthorship, or endorsement of the conclusions.

References, third-party software, and datasets retain their original attribution. The supplied reports are preserved unchanged, including their printed dates. Study dates are January 2026 (molecular potentials), March 2026 (electronic states), and May 2026 (diagnostics). Technical environment records and random seeds are preserved as supplied.
