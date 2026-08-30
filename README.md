# Updatable lattice-based signatures

Python proof of concept developed for the master's thesis *Implementacion de
protocolo actualizable de firma digital basado en reticulos*.

The repository contains an experimental implementation of the USGPV updatable
signature proposal, a CRSST21-based comparison implementation, a simplified
verifiable-encryption substitute, and scripts used to study parameter growth,
performance, and statistical behaviour.

A detailed description of the modules and execution flow is available in
[`ARQUITECTURA.md`](ARQUITECTURA.md).

> [!WARNING]
> This is academic research software. It has not been security-audited, is not
> constant-time, contains simplified or substitute components, and must not be
> used to protect real information or in production systems.

## Contents

- `src/usgpv.py`: main USGPV proof-of-concept implementation and the simulation
  and statistical-analysis functions used in the thesis.
- `src/crsst21.py`: adapted CRSST21 implementation used as an experimental
  comparison baseline.
- `src/ve_impostor.py`: simplified encryption substitute used by the prototype.
  It is not an implementation of a production-ready verifiable-encryption
  scheme.
- `src/correccion_gaussiana.py`: two-dimensional demonstration of Gaussian correction noise
  during signature updates.
- `crecimiento_dispersion.py`: plots the evolution of the dispersion parameter across update
  epochs.

## Requirements

- Python 3.10 or later
- NumPy
- Matplotlib
- psutil
- PyNaCl

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Running the prototype

Use the central launcher from the repository root:

```bash
python ejecutar.py --help
python ejecutar.py demo
```

Available commands are:

- `demo`: short USGPV sign, verify, update, and verify workflow.
- `rendimiento-usgpv`: monitored USGPV simulation.
- `rendimiento-crsst21`: monitored CRSST21 comparison simulation.
- `covarianza`: effective-dispersion statistical analysis.
- `pca`: PCA comparison between USGPV and CRSST21.
- `correccion-gaussiana`: two-dimensional Gaussian-correction visualization.
- `graficas-dispersion`: dispersion-growth plots.

Each launcher can also be run independently as a module, for example:

```bash
python -m scripts.demostracion_usgpv --message "mensaje de prueba"
python -m scripts.demostracion_correccion_gaussiana --samples 5000
python -m scripts.graficas_dispersion
```

La demostración de la corrección gaussiana utiliza muestras discretas por
defecto. Para visualizar la versión continua se puede añadir `--continuous`.


After installing the project with `python -m pip install .`, the same operations
are available as commands such as `tfm-signatures`, `tfm-usgpv-demo`,
`tfm-pca`, and `tfm-parameter-plots`.

Some simulations are computationally expensive and use large in-memory NumPy
matrices. Start with small dimensions and sample sizes when adapting the code.

## Reproducibility

The experiments described in the thesis use fixed seeds in selected simulation
functions, including seed `2026`. A fixed pseudo-random seed improves experiment
reproducibility but does not provide cryptographically secure randomness.

Representative thesis parameters include `n = 32`, five epochs, `q = 128`, and
`p0 = 0.3`. Large dispersion parameters can grow rapidly across epochs and may
eventually cause numerical overflow. This is a documented limitation of the
proof of concept.

## Academic scope

The implementation approximates theoretical lattice samplers and includes a
simplified encryption component. Experimental results therefore apply only to
the simulated conditions and do not establish implementation-level security.

## Citation

Citation metadata is available in `CITATION.cff`. GitHub can use this file to
display a **Cite this repository** option.

## License

Copyright (C) 2026 Juan José Campos Rosa.

The source code in this repository is licensed under the GNU Affero General
Public License, version 3 or any later version (`AGPL-3.0-or-later`). Commercial
use is permitted, but copyright and license notices must be preserved. Modified
versions that are distributed, or made available for users to interact with
over a network, must provide the corresponding source code under the AGPL.

The thesis document is separate academic material and retains the Creative
Commons license stated inside that document. Its own license notice takes
precedence for the PDF.
