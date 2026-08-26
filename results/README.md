# Results

Everything here is produced by `src/pipeline.py`. Nothing is hand-edited.

Running `python src/pipeline.py reproduce` rewrites the starred rows below from
`scores/*.npz` in about fifteen seconds, and they should come back identical to
what is committed.

| Path | Produced by | Contents |
|---|---|---|
| `scores/{ds}_seed{n}.npz` | `train` | calibrated probabilities and labels, calibration and test blocks |
| `tables/paper_A_coverage.csv` * | `reproduce` | marginal against class-conditional coverage — **Table 3** |
| `tables/paper_C_price.csv` * | `reproduce` | alert volume against target fraud coverage — **Table 4** |
| `tables/paper_D_ncal_sweep.csv` * | `reproduce` | price and degeneracy against calibration fraud count — **Table 5** |
| `tables/paper_E_gap.csv` * | `reproduce` | the coverage dissociation, per seed — **Figure 2** |
| `tables/paper_wilcoxon.csv` * | `reproduce` | paired Wilcoxon on per-seed cost, Holm-corrected — **Table B.2** |
| `tables/paper_models.csv` | `train` | the three detectors, per seed — **Table B.1** |
| `tables/paper_ablation.csv` | `train` | stage-wise ablation, raw cost — **Table B.3** |
| `tables/paper_gate.csv` | `train` | learned blend weights, per seed — Section 7.1 |
| `tables/table02_splits_*.csv` | `train` | block sizes and fraud counts — **Table 2** |
| `all_results.json` | `train` | full per-seed record: splits, gate, models, ablation, SemiSync, conformal |
| `PAPER_RESULTS.md` | `train` | narrative dump of the above |

Figures land in `../figures/` and map to the paper as:

| File | Paper |
|---|---|
| `fig_coverage_gap.png` | Figure 2 — marginal against fraud-class coverage |
| `fig_mechanism.png` | Figure 3 — the two score CDFs, one seed |
| `fig_coverage_price.png` | Figure 4 — alert volume against target coverage |
| `fig_ncal_price.png` | Figure 5 — price against calibration fraud count |

Figure 1 of the paper is the protocol schematic and is drawn in the manuscript,
not here.

## Naming bridge to the paper

Generated files keep the pipeline's internal names. Read them as:
`T` = ULB, `K` = Sparkov, `PS` = PaySim; `ADAPTIVE-CP-FRAUD` = the
cost-sensitive detector, `HybridMeta-XGB` = **Hybrid-XGB** in the paper;
ablation configs `full` / `no_stage1_uniform` / `no_stage2_spw` /
`baseline_no_anomaly` = full pipeline / no learned weights / no cost gradient
(SPW) / no anomaly features (= Baseline SPW).

One column is deliberately not in the paper: `all_results.json` records a
`"psd2"` block computed under the **marginal** conformal rule. The paper's
fraud rate among approved transactions is the class-conditional one, in
`paper_C_price.csv` as `lambda_approved_pct`. Do not compare the two.

## The .npz files are the point

They hold the calibrated probabilities, so every conformal result in the paper
can be recomputed without retraining and without downloading any dataset:

```python
import numpy as np
z = np.load("results/scores/T_seed42.npz")
p_cal, y_cal, p_test, y_test = z["p_cal"], z["y_cal"], z["p_test"], z["y_test"]
```

`python src/pipeline.py reproduce` does exactly this for all thirteen files. A
reviewer can check Tables 3, 4, 5 and B.2 and all four data figures in under a
minute, on a laptop, offline.
