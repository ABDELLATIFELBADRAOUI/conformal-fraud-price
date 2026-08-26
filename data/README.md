# Datasets

None is redistributed here. Download the four CSVs, put them in this folder (or
point `$CFRAUD_DATA_DIR` at wherever they live), and run
`python src/pipeline.py train`.

Nothing in this section is needed to check the paper's conformal results:
`python src/pipeline.py reproduce` works from `results/scores/*.npz` alone.

## ULB Credit Card Fraud

<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

`creditcard.csv` — 284,807 European transactions, September 2013, 492 frauds
(0.172 %). V1–V28 are PCA projections; we add `log(1+Amount)` and a normalised
elapsed-time column. Sorted by `Time` before splitting.

The elapsed-time column is retained. `Time` is seconds since the first
transaction, so under the chronological split the test block sits at the top of
the range and the extrapolation concern stated for Sparkov's `unix_time`
applies here in weaker form. Dividing by the global maximum is inert: the
train-block standardisation that follows undoes any fixed positive rescaling of
a column exactly.

## Sparkov

<https://www.kaggle.com/datasets/kartik2112/fraud-detection>

`fraudTrain.csv` and `fraudTest.csv` — 1,296,675 and 555,719 simulated card
transactions, 7,506 and 2,145 frauds. The provider's own chronological split is
used: train, tune and calibration are the 55/15/15 leading portions of
`fraudTrain`, and `fraudTest` is the test block, so the trailing 15 % of
`fraudTrain` is unused. Features: transaction amount, cardholder and merchant
coordinates, city population, ZIP code, the Unix timestamp, and an ordinal
encoding of the merchant category. Sorted by `unix_time`.

## PaySim

<https://www.kaggle.com/datasets/ealaxi/paysim1>

`paysim.csv` — 6,362,620 simulated mobile-money transfers, 8,213 frauds
(0.129 %). Sorted by `step`. Rename the downloaded file to `paysim.csv`, or set
`CFG["FILE_PS"]`.

Three columns are deliberately excluded, and the exclusions matter:

- **`isFlaggedFraud`** is the simulator's own rule flag. Including it hands the
  model a partial oracle.
- **`step`** orders the rows but is not a feature. A temporal index under a
  chronological split invites the model to learn position in time rather than
  fraud.
- **`nameOrig` / `nameDest`** are near-unique identifiers. Only the useful part
  is kept: whether the destination is a merchant.

Retained: amount and its log, the four balance fields, two derived balance
inconsistencies, the merchant-destination indicator, and one-hot transaction
types — 14 features.

**Caveat.** PaySim fraud is injected by rule and empties the originating
account, so `errBalanceOrig` is close to deterministic on fraudulent transfers.
The detector reaches AUPRC 0.989 there, far above real card data. The paper
uses PaySim for its high-separability regime and makes no claim about detection
performance on it.

## Checksums

These are the four files that produced the released results.

```bash
sha256sum creditcard.csv fraudTrain.csv fraudTest.csv paysim.csv
```

| File | Bytes | SHA-256 |
|---|---:|---|
| `creditcard.csv` | 150,828,752 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| `fraudTrain.csv` | 351,238,196 | `fd7139200dbfcbed0b6742bbe05a4f1abce532c4fef20918228a651647a3e75d` |
| `fraudTest.csv` | 150,354,339 | `12d553ab19440c752d2531ee1af44bb64f12cc3d3839f1649f19e81c230545f0` |
| `paysim.csv` | 493,534,783 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |

A mismatch usually means Kaggle has re-exported the file, not that anything is
wrong; the row counts above are the check that matters.

## Resulting block sizes

Rows / frauds / rate.

| Dataset | train | tune | calibration | test |
|---|---|---|---|---|
| ULB | 156,643 / 350 / 0.223 % | 42,721 / 34 / 0.080 % | 42,721 / 56 / 0.131 % | 42,722 / 52 / 0.122 % |
| Sparkov | 713,171 / 4,226 / 0.593 % | 194,501 / 895 / 0.460 % | 194,501 / 1,252 / 0.644 % | 555,719 / 2,145 / 0.386 % |
| PaySim | 3,499,441 / 2,905 / 0.083 % | 954,393 / 738 / 0.077 % | 954,393 / 562 / 0.059 % | 954,393 / 4,008 / 0.420 % |

The fraud rate moves by a factor of seven between PaySim's calibration and test
blocks under a strictly chronological split. This label shift is what the
conformal guarantee has to absorb, and Section 8.3 of the paper shows that the
Kolmogorov–Smirnov term does not detect it.

## IEEE-CIS

`train_transaction.csv` is **not** used. At roughly 28 h per seed under this
pipeline, ten seeds would take eleven days; the compute went to more seeds on
the two card benchmarks instead. `load_V` in `src/pipeline.py` is kept only so
the exclusion is legible.
