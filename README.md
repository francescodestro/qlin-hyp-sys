# qlin-hyp-sys

Code for "A hybrid characteristics/lines numerical method for quasilinear hyperbolic systems" (Destro, Salvador-Pomarol, Braatz).

## Contents

- `numerics.py` -- solver (SSPRK(3,3), adaptive step control, moving mesh)
- `manuscript_helpers.py` -- case study definitions and plotting
- `generate_results.ipynb` -- reproduces all figures and tables
- `case_study_outputs/` -- precomputed results

## Usage

Run `generate_results.ipynb` with Python 3.10+ and NumPy, SciPy, Matplotlib, Pandas.
