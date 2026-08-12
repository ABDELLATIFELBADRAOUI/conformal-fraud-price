# The price of a conformal coverage guarantee in card fraud detection

Code and results for ***Conformal Coverage Guarantees Do Not Reduce the
Recall–Alert-Volume Trade-off in Card Fraud Detection*** (submitted).

## The result in one table

A split conformal predictor at level $\alpha$ guarantees the true label lies in
the prediction set with probability $\geq 1-\alpha$. In fraud detection this is
routinely read as a bound on missed frauds. It is not: the marginal guarantee
constrains a mixture dominated by the legitimate class. The construction that
does deliver the intended guarantee — one quantile per class — has a price, and
the price depends on how well the detector separates the two classes.

| | Detector AUPRC | Fraud-class coverage at 0.95 target | Alert volume |
|---|---|---|---|
| PaySim | 0.989 | 0.957 | **0.41 %** |
| ULB | 0.712 | 1.000 | **100 %** |
| Sparkov | 0.529 | 0.974 | **73.6 %** |

Across all three datasets and every level tested, the marginal predictor meets
its own target while covering between 0.009 and 0.452 of frauds.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/ADAPTIVE_CP_FRAUD_v12.ipynb
```

Set `CFG["DATA_DIR"]` in cell 1 to the folder holding the CSVs (see
`data/README.md`). Cell 1 checks every file is present before any long
computation starts.

### Two ways in

**Fast path — reuse the released scores.** `results/scores/*.npz` holds the
calibrated probabilities on the calibration and test blocks for every dataset
and seed. Cells 12, 13 and 14 recompute **every conformal result, table and
figure in the paper** from these, in seconds, with no training and no dataset
download. This is the path to check our numbers.

**Full path — retrain.** Budget roughly 10 min per seed on ULB, 30 min on
Sparkov and 3 h on PaySim, on the hardware in the paper, with `N_JOBS = 1`.

## Protocol

Each dataset is split chronologically into four contiguous blocks:

| Block | Share | Used for |
|---|---|---|
| train | 55 % | one-class detectors, boosted trees |
| tune | 15 % | blend weights, early stopping, calibrator fit, threshold |
| **calibration** | 15 % | **conformal quantile, and nothing else** |
| test | 15 % | evaluation, once |

Separating *tune* from *calibration* is what makes the coverage statement
applicable. A single held-out block serving both purposes — the common
three-way practice — computes the quantile on data the predictor has already
been selected against.

## Determinism

Sub-sampling for LOF and One-Class SVM draws from a generator derived from
`sha256(seed:stage)`, never from the global NumPy state, so results do not
depend on cell execution order. With `N_JOBS = 1` the pipeline reproduces the
reported digits; with `N_JOBS = -1` it is faster but multi-threaded
floating-point reduction order is not fixed.

## Seed counts

Detector and ablation results use ten seeds on ULB and Sparkov and three on
PaySim. The conformal analyses use the five ULB, five Sparkov and three PaySim
seeds whose score vectors were retained and are released here.

## Layout

```
notebooks/ADAPTIVE_CP_FRAUD_v12.ipynb   annotated pipeline — start here
src/pipeline.py                         the same code as a flat module
data/README.md                          download instructions and checksums
results/scores/*.npz                    calibrated probabilities per dataset and seed
results/tables/*.csv                    one CSV per table in the paper
results/PAPER_RESULTS.md                every number in the paper, with its source
figures/                                the four figures
```

## Not in this repository

Datasets are redistributed by neither Kaggle's terms nor ours. See
`data/README.md`.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`. The datasets carry their own terms.
