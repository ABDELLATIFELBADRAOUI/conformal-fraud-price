# The price of a conformal coverage guarantee in card fraud detection

Code and results for ***Quantifying the Empirical Alert-Volume Cost of
Class-Conditional Conformal Prediction for Card Fraud Detection Under the
Revised EU Payment Services Directive***.

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
| ULB | 0.713 | 1.000 | **100 %** |
| Sparkov | 0.529 | 0.974 | **73.6 %** |

Across all three datasets and every level tested, the marginal predictor meets
its own target while covering between 0.009 and 0.452 of frauds. The ULB row is
the degenerate flag-everything rule.

## Verifying, without downloading anything

```bash
pip install numpy pandas scipy matplotlib
python src/pipeline.py reproduce
```

About fifteen seconds. `results/scores/*.npz` holds the calibrated
probabilities on the calibration and test blocks for every dataset and seed, so
this recomputes **every conformal table and figure in the paper** — Tables 3, 4
and B.2, and Figures 2 to 5 — with no training, no dataset download and no
xgboost. It rewrites `results/tables/*.csv` and `figures/*.png`; both should
come back identical to what is committed here.

## Retraining from the raw data

```bash
pip install -r requirements.txt
export CFRAUD_DATA_DIR=/path/to/the/four/csv/files    # see data/README.md
python src/pipeline.py train
```

Budget roughly 10 min per seed on ULB, 30 min on Sparkov and 3 h on PaySim, on
the hardware in the paper, with `N_JOBS = 1`.

`notebooks/ADAPTIVE_CP_FRAUD_v12.ipynb` is the annotated version of the same
pipeline, cell by cell, and is the form in which the reported results were
produced.

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

Sparkov is the exception: it uses the provider's own chronological split, so
the 55/15/15 leading portions of `fraudTrain.csv` supply train, tune and
calibration, and `fraudTest.csv` is the test block.

## Determinism

Sub-sampling for LOF and One-Class SVM draws from a generator derived from
`sha256(seed:stage)`, never from the global NumPy state, so results do not
depend on execution order. `python src/pipeline.py selftest` checks this. With
`N_JOBS = 1` the pipeline reproduces the reported digits; with `N_JOBS = -1` it
is faster but the multi-threaded floating-point reduction order is not fixed.

## Seed counts

Detector and ablation results use ten seeds on ULB and Sparkov and three on
PaySim. The conformal analyses use the five ULB, five Sparkov and three PaySim
seeds whose score vectors were retained and are released here.

## Layout

```
src/pipeline.py                         the pipeline; importable, no side effects
notebooks/ADAPTIVE_CP_FRAUD_v12.ipynb   the same code annotated, cell by cell
data/README.md                          download instructions and exclusions
results/scores/*.npz                    calibrated probabilities per dataset and seed
results/tables/*.csv                    one CSV per table in the paper
results/PAPER_RESULTS.md                every number in the paper, with its source
figures/                                the four data figures of the paper
```

Paths are configurable through `$CFRAUD_DATA_DIR` and `$CFRAUD_OUT_DIR`; the
defaults are `data/` and `results/` inside this repository.

## Not in this repository

The datasets. Neither Kaggle's terms nor ours redistribute them. See
`data/README.md`.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`. The datasets carry their own terms.
