# Results

Everything here is produced by the notebook. Nothing is hand-edited.

| Path | Produced by | Contents |
|---|---|---|
| `PAPER_RESULTS.md` | cell 12 | every number in the paper, with the table it belongs to |
| `all_results.json` | cells 10–11ter | full per-seed record: splits, gate, models, ablation, SemiSync, conformal |
| `scores/{ds}_seed{n}.npz` | cell 9 | calibrated probabilities and labels, calibration and test blocks |
| `tables/paper_A_coverage.csv` | cell 12 | marginal against class-conditional coverage |
| `tables/paper_C_price.csv` | cell 12 | alert volume against target fraud coverage |
| `tables/paper_D_ncal_sweep.csv` | cell 13 | price and degeneracy against calibration fraud count |
| `tables/paper_E_gap.csv` | cell 14 | the coverage dissociation, per seed |
| `tables/paper_models.csv` | cell 12 | the three detectors, per seed |
| `tables/paper_ablation.csv` | cell 12 | stage-wise ablation, raw cost |
| `tables/paper_gate.csv` | cell 12 | learned blend weights, per seed |
| `tables/table02_splits_*.csv` | cells 10–11ter | block sizes and fraud counts |

Figures, in `../figures/`: `fig_coverage_gap.png` (Figure 1),
`fig_coverage_price.png` (Figure 3), `fig_ncal_price.png` (Figure 4),
`fig_mechanism.png` (supporting).

## The .npz files are the point

They hold the calibrated probabilities, so every conformal result in the paper
can be recomputed without retraining and without downloading any dataset:

```python
import numpy as np
z = np.load("results/scores/T_seed42.npz")
p_cal, y_cal, p_test, y_test = z["p_cal"], z["y_cal"], z["p_test"], z["y_test"]
```

Cells 12, 13 and 14 do exactly this. A reviewer can check Tables 2 and 3 and
all four figures in about a minute, on a laptop, offline.
