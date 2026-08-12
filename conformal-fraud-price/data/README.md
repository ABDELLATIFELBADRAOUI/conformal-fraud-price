# Datasets

None is redistributed here. Download the four CSVs, place them in this folder,
and set `CFG["DATA_DIR"]` in cell 1 of the notebook.

## ULB Credit Card Fraud

<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

`creditcard.csv` — 284,807 European transactions, September 2013, 492 frauds
(0.172 %). V1–V28 are PCA projections; we add `log(1+Amount)` and a normalised
timestamp. Sorted by `Time` before splitting.

## Sparkov

<https://www.kaggle.com/datasets/kartik2112/fraud-detection>

`fraudTrain.csv` and `fraudTest.csv` — 1,296,675 and 555,719 simulated card
transactions, 7,506 and 2,145 frauds. The four blocks are carved from
`fraudTrain`; `fraudTest` serves as the test block. Features: merchant
category, amount, cardholder and merchant coordinates. Sorted by `unix_time`.

## PaySim

<https://www.kaggle.com/datasets/ealaxi/paysim1>

`paysim.csv` — 6,362,620 simulated mobile-money transfers, 8,213 frauds
(0.129 %). Sorted by `step`.

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

Verify after download and record them here:

```
sha256sum creditcard.csv fraudTrain.csv fraudTest.csv paysim.csv
```

| File | SHA-256 |
|---|---|
| creditcard.csv | `[TO FILL]` |
| fraudTrain.csv | `[TO FILL]` |
| fraudTest.csv | `[TO FILL]` |
| paysim.csv | `[TO FILL]` |

## Resulting block sizes

Rows / frauds / rate.

| Dataset | train | tune | calibration | test |
|---|---|---|---|---|
| ULB | 156,643 / 350 / 0.223 % | 42,721 / 34 / 0.080 % | 42,721 / 56 / 0.131 % | 42,722 / 52 / 0.122 % |
| Sparkov | 713,171 / 4,226 / 0.593 % | 194,501 / 895 / 0.460 % | 194,501 / 1,252 / 0.644 % | 555,719 / 2,145 / 0.386 % |
| PaySim | 3,499,441 / 2,905 / 0.083 % | 954,393 / 738 / 0.077 % | 954,393 / 562 / 0.059 % | 954,393 / 4,008 / 0.420 % |

The fraud rate moves by a factor of seven between PaySim's calibration and test
blocks under a strictly chronological split. This label shift is what the
conformal guarantee has to absorb, and Section 6.4 of the paper shows that the
Kolmogorov–Smirnov inflation term does not detect it.
