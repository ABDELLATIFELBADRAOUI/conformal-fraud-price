# ============================================================================
#  conformal-fraud-price — reproduction pipeline
#
#  Companion code for "Quantifying the Empirical Alert-Volume Cost of
#  Class-Conditional Conformal Prediction for Card Fraud Detection Under the
#  Revised EU Payment Services Directive".
#
#  Two entry points, and only the first one needs anything but this file:
#
#    python src/pipeline.py reproduce
#        Recomputes every conformal table and figure in the paper from the
#        released per-seed score vectors in results/scores/*.npz. No dataset
#        download, no retraining, no xgboost. Seconds on a laptop.
#
#    python src/pipeline.py train
#        Re-runs the full experiment from the raw CSVs. Requires the four
#        Kaggle files, xgboost, and days of compute.
#
#  Importing this module has no side effects: nothing is trained, no directory
#  is created and no path is checked until you call something.
#
#  Paths are resolved in this order:
#      $CFRAUD_DATA_DIR  ->  <repo>/data       (raw CSVs, for `train`)
#      $CFRAUD_OUT_DIR   ->  <repo>/results    (outputs, and where the
#                                               released .npz already live)
#
#  MIT licence. See LICENSE.
# ============================================================================
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
#  Paths and configuration
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v).expanduser() if v else default


CFG = dict(
    # --- raw CSV file names (all four expected in one directory) -------------
    FILE_T       = "creditcard.csv",   # ULB
    FILE_K_TRAIN = "fraudTrain.csv",   # Sparkov, provider split
    FILE_K_TEST  = "fraudTest.csv",    # Sparkov, provider split
    FILE_PS      = "paysim.csv",       # PaySim
    FILE_V       = "train_transaction.csv",   # IEEE-CIS, excluded (see below)

    # --- four-way chronological split ---------------------------------------
    #   train : detectors and booster
    #   val   : blend weights, early stopping, probability calibrator, tau*
    #   cal   : the conformal quantile and nothing else
    #   test  : evaluated once
    FRAC_TRAIN=0.55, FRAC_VAL=0.15, FRAC_CAL=0.15,   # test = remainder

    # --- costs ---------------------------------------------------------------
    C_FP=1.0, C_FN=10.0,

    # --- anomaly detectors ---------------------------------------------------
    IF_TREES=200, LOF_K=20, OCSVM_NU=0.002,
    LOF_CAP=50_000, OCSVM_CAP=20_000,
    CONTAM_INIT=0.002, CONTAM_LO=0.001, CONTAM_HI=0.10,

    # --- XGBoost -------------------------------------------------------------
    XGB=dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
             min_child_weight=5),
    EARLY_STOP=50,

    # --- blend-weight search -------------------------------------------------
    NM_STARTS=10, NM_MAXEVAL=1000, N_TAU_GRID=301,

    # --- SemiSync ------------------------------------------------------------
    SS_MAX_ITER=10, SS_TOL=1e-3, SS_CONF=0.95,

    # --- conformal -----------------------------------------------------------
    ALPHA_SIG=[0.05, 0.10, 0.20],

    # --- reproducibility -----------------------------------------------------
    # IEEE-CIS is excluded: roughly 28 h per seed under this pipeline, which
    # ten seeds would turn into eleven days. The compute goes to more seeds on
    # the two card benchmarks instead.
    SEEDS=list(range(42, 52)),   # PaySim uses SEEDS_PS below
    SEEDS_PS=[42, 43, 44],
    N_JOBS=1,   # 1 = bit-reproducible. -1 is faster, but the order of the
                # multi-threaded float reductions is then not fixed.
)

CFG["DATA_DIR"] = str(_env_path("CFRAUD_DATA_DIR", REPO_ROOT / "data"))
CFG["OUT_DIR"] = str(_env_path("CFRAUD_OUT_DIR", REPO_ROOT / "results"))

K_MODE = "provider"   # "provider" = fraudTrain / fraudTest, as the paper does
                      # "single"   = fraudTrain alone, re-split


def data_path(key: str, cfg=CFG) -> Path:
    """Absolute path of one raw CSV. Nothing is read or checked here."""
    return Path(cfg["DATA_DIR"]) / cfg[key]


def out_dir(cfg=CFG) -> Path:
    return Path(cfg["OUT_DIR"])


def _ensure_out(cfg=CFG) -> Path:
    o = out_dir(cfg)
    (o / "tables").mkdir(parents=True, exist_ok=True)
    (o / "scores").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    return o


def check_data(cfg=CFG, keys=("FILE_T", "FILE_K_TRAIN", "FILE_K_TEST")) -> list:
    """Return the names of the raw files that are missing. Never raises."""
    missing = []
    for k in keys:
        p = data_path(k, cfg)
        if not p.exists():
            missing.append(p.name)
    return missing


def _require_xgboost():
    """xgboost is needed only by `train`; `reproduce` runs without it."""
    try:
        import xgboost as xgb
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "xgboost is required to retrain the models "
            "(pip install xgboost). It is NOT required to reproduce the "
            "paper's conformal results from results/scores/*.npz."
        ) from exc
    return xgb


# ============================================================================
#  Deterministic generators
# ============================================================================
def rng_for(seed, name):
    """Generator that does not depend on execution order."""
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def subsample_idx(n, cap, seed, name):
    """Stable subsampling indices, without replacement."""
    if n <= cap:
        return np.arange(n)
    return np.sort(rng_for(seed, name).choice(n, cap, replace=False))


def selftest_determinism() -> bool:
    """Two calls agree whatever happens between them."""
    a = subsample_idx(100_000, 20_000, 42, "ocsvm")
    rng_for(42, "something_else").random(10_000)          # interleaved noise
    b = subsample_idx(100_000, 20_000, 42, "ocsvm")
    return bool(np.array_equal(a, b))


# ============================================================================
#  Data loading and the four-way chronological split
# ============================================================================
def four_way(X, y, cfg=CFG):
    """Chronological split into train / val / cal / test.

    The scaler is fitted on the train block only.
    """
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    c1 = int(n * cfg["FRAC_TRAIN"])
    c2 = c1 + int(n * cfg["FRAC_VAL"])
    c3 = c2 + int(n * cfg["FRAC_CAL"])
    sl = dict(train=slice(0, c1), val=slice(c1, c2),
              cal=slice(c2, c3), test=slice(c3, n))
    out = {k: (X[s], y[s]) for k, s in sl.items()}
    sc = StandardScaler().fit(out["train"][0])
    return {k: (sc.transform(Xk), yk) for k, (Xk, yk) in out.items()}


def report_split(name, S, verbose=True):
    rows = []
    for k in ("train", "val", "cal", "test"):
        Xk, yk = S[k]
        rows.append(dict(dataset=name, split=k, n=len(yk),
                         frauds=int(yk.sum()),
                         rate_pct=round(100 * yk.mean(), 4)))
    df = pd.DataFrame(rows)
    if verbose:
        print(df.to_string(index=False))
    return df


def load_T(path):
    """ULB. Note that a normalised elapsed-time column IS retained.

    `Time` is seconds since the first transaction in the file, so under the
    chronological split the test block occupies the top of the range and the
    same extrapolation caveat that the paper states for Sparkov's raw
    `unix_time` applies here in weaker form. Dividing by the global maximum is
    inert: `four_way` then standardises on the train block, and a fixed
    positive rescaling of a column is undone exactly by that standardisation.
    """
    df = pd.read_csv(path)
    df = df.sort_values("Time").reset_index(drop=True)
    X = df.drop(columns=["Class", "Amount", "Time"]).copy()
    X["log_amount"] = np.log1p(df["Amount"].values)
    X["time_norm"] = df["Time"].values / df["Time"].max()
    return X.values.astype(np.float32), df["Class"].values.astype(int)


def load_K(path_train, path_test, mode=K_MODE):
    """Sparkov. The raw `unix_time` is retained; the paper states the asymmetry
    with PaySim's excluded `step` rather than resolving it."""
    NUM = ["amt", "lat", "long", "merch_lat", "merch_long",
           "city_pop", "unix_time", "zip"]

    def feats(df, cat_map=None):
        cols = [c for c in NUM if c in df.columns]
        d = df.copy()
        if "category" in d.columns:
            if cat_map is None:
                cs = d["category"].astype("category")
                cat_map = {c: i for i, c in enumerate(cs.cat.categories)}
            d["cat_ord"] = d["category"].map(cat_map).fillna(-1).astype(int)
            cols = cols + ["cat_ord"]
        lbl = "is_fraud" if "is_fraud" in d.columns else "isFraud"
        return (d[cols].fillna(0).values.astype(np.float32),
                d[lbl].astype(int).values, cat_map)

    tr = pd.read_csv(path_train)
    tcol = "unix_time" if "unix_time" in tr.columns else "trans_date_trans_time"
    tr = tr.sort_values(tcol).reset_index(drop=True)
    Xtr, ytr, cmap = feats(tr)
    if mode == "single":
        return Xtr, ytr, None
    te = pd.read_csv(path_test).sort_values(tcol).reset_index(drop=True)
    Xte, yte, _ = feats(te, cmap)
    return Xtr, ytr, (Xte, yte)


def load_PS(path, max_rows=None):
    """PaySim — simulated mobile-money transfers.

    Exclusions, and why they matter:

    * `isFlaggedFraud` is the simulator's own rule flag (it fires on TRANSFERs
      above a threshold). Including it hands the model a partial oracle.
    * `step` orders the rows but is not a feature. Under a chronological split
      whose test block is later, a temporal index invites the model to learn
      position in time rather than fraud.
    * `nameOrig` / `nameDest` are near-unique identifiers. Only the useful part
      is kept: whether the destination is a merchant (prefix M).
    * `type` is one-hot encoded: fraud appears in only two of the five types,
      which is the strongest signal in the dataset.
    * The two balance discrepancies are derived because they encode the
      accounting inconsistency that characterises fraudulent transfers.
    """
    df = pd.read_csv(path)
    need = {"step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFraud"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"PaySim columns missing: {sorted(missing)}; "
                         f"present: {list(df.columns)}")

    df = df.sort_values("step", kind="mergesort").reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)

    y = df["isFraud"].astype(int).values

    X = pd.DataFrame(index=df.index)
    X["amount"] = df["amount"].astype(np.float32)
    X["log_amount"] = np.log1p(df["amount"].values).astype(np.float32)
    X["oldbalanceOrg"] = df["oldbalanceOrg"].astype(np.float32)
    X["newbalanceOrig"] = df["newbalanceOrig"].astype(np.float32)
    X["oldbalanceDest"] = df["oldbalanceDest"].astype(np.float32)
    X["newbalanceDest"] = df["newbalanceDest"].astype(np.float32)
    X["errBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]
                           - df["oldbalanceOrg"]).astype(np.float32)
    X["errBalanceDest"] = (df["oldbalanceDest"] + df["amount"]
                           - df["newbalanceDest"]).astype(np.float32)
    if "nameDest" in df.columns:
        X["dest_is_merchant"] = (df["nameDest"].astype(str)
                                 .str.startswith("M").astype(np.float32))
    for t in sorted(df["type"].astype(str).unique()):
        X[f"type_{t}"] = (df["type"].astype(str) == t).astype(np.float32)
    return X.values.astype(np.float32), y


def load_V(path):
    """IEEE-CIS loader. Kept for completeness; the dataset is NOT used in the
    paper (roughly 28 h per seed under this pipeline)."""
    df = pd.read_csv(path)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    y = df["isFraud"].astype(int).values
    drop = (["isFraud", "TransactionID", "TransactionDT", "ProductCD",
             "card4", "card6", "P_emaildomain", "R_emaildomain"]
            + [f"M{i}" for i in range(1, 10)])
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    return X.values.astype(np.float32), y


# ============================================================================
#  Stage 0 — unsupervised anomaly features
# ============================================================================
def stage0(S, seed, contamination=None, cfg=CFG):
    """Return {split: (a_if, a_lof, a_oc)}, rank-normalised into [0, 1]."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import QuantileTransformer

    cont = cfg["CONTAM_INIT"] if contamination is None else contamination
    Xtr, ytr = S["train"]
    X0 = Xtr[ytr == 0]                                   # legitimate only

    iso = IsolationForest(n_estimators=cfg["IF_TREES"], contamination=cont,
                          random_state=seed, n_jobs=cfg["N_JOBS"]).fit(X0)

    i_lof = subsample_idx(len(X0), cfg["LOF_CAP"], seed, "lof")
    lof = LocalOutlierFactor(n_neighbors=cfg["LOF_K"], novelty=True,
                             n_jobs=cfg["N_JOBS"]).fit(X0[i_lof])

    i_oc = subsample_idx(len(X0), cfg["OCSVM_CAP"], seed, "ocsvm")
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=cfg["OCSVM_NU"]).fit(X0[i_oc])

    raw = {k: (-iso.score_samples(S[k][0]),
               -lof.score_samples(S[k][0]),
               -oc.score_samples(S[k][0])) for k in S}

    out = {k: [] for k in S}
    for j in range(3):                                   # QT fitted on train
        qt = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=min(1000, len(S["train"][1])),
                                 random_state=seed)
        qt.fit(raw["train"][j].reshape(-1, 1))
        for k in S:
            out[k].append(qt.transform(raw[k][j].reshape(-1, 1)).ravel())
    return {k: tuple(v) for k, v in out.items()}


# ============================================================================
#  Stages 1 and 2 — learned blend weights, cost-sensitive gradient
# ============================================================================
def expected_cost(scores, y, cfp=None, cfn=None, n_tau=None, cfg=CFG):
    """Expected cost integrated over the threshold grid."""
    cfp = cfg["C_FP"] if cfp is None else cfp
    cfn = cfg["C_FN"] if cfn is None else cfn
    n_tau = cfg["N_TAU_GRID"] if n_tau is None else n_tau
    pi1 = y.mean(); pi0 = 1 - pi1
    n0 = max((y == 0).sum(), 1); n1 = max((y == 1).sum(), 1)
    tot = 0.0
    for t in np.linspace(0, 1, n_tau):
        pred = scores >= t
        fp = np.count_nonzero(pred & (y == 0))
        fn = np.count_nonzero(~pred & (y == 1))
        tot += cfp * (fp / n0) * pi0 + cfn * (fn / n1) * pi1
    return float(tot / n_tau)


def blend(alpha, trio):
    a = np.clip(alpha, 0, None)
    a = a / (a.sum() + 1e-12)
    return a[0] * trio[0] + a[1] * trio[1] + a[2] * trio[2]


def adaptive_gate(trio_val, y_val, seed, cfg=CFG):
    """Stage 1: alpha* minimising the integrated expected cost ON THE VAL
    BLOCK. The calibration block is never touched here."""
    from scipy.optimize import minimize
    obj = lambda al: expected_cost(blend(al, trio_val), y_val)   # noqa: E731
    best, best_ec, ecs = None, np.inf, []
    starts = [np.array([1 / 3, 1 / 3, 1 / 3])]
    g = rng_for(seed, "nelder_mead")
    starts += [g.dirichlet([1, 1, 1]) for _ in range(cfg["NM_STARTS"] - 1)]
    for s0 in starts:
        r = minimize(obj, s0, method="Nelder-Mead",
                     options=dict(maxfev=cfg["NM_MAXEVAL"],
                                  xatol=1e-4, fatol=1e-6))
        a = np.clip(r.x, 0, None)
        a = a / (a.sum() + 1e-12)
        ec = obj(a)
        ecs.append(ec)
        if ec < best_ec:
            best, best_ec = a, ec
    ec_unif = obj(np.array([1 / 3, 1 / 3, 1 / 3]))
    return dict(alpha=np.round(best, 6), ec_star=best_ec, ec_uniform=ec_unif,
                gain_pct=100 * (ec_unif - best_ec) / ec_unif,
                ec_var_across_starts=float(np.var(ecs)))


def cost_objective(cfp, cfn):
    """Stage 2: exact gradient, strictly positive surrogate second-order term."""
    from scipy.special import expit

    def obj(pred, dtrain):
        y = dtrain.get_label()
        p = expit(pred)
        w = p * (1 - p)
        grad = w * (cfp * (1 - y) - cfn * y)
        surrogate_hessian = w * (cfp * (1 - y) + cfn * y) + 1e-6
        return grad, surrogate_hessian
    return obj


def augment(X, trio, alpha):
    return np.column_stack([X, trio[0], trio[1], trio[2], blend(alpha, trio)])


def train_booster(Xtr, ytr, Xval, yval, seed, use_cost=True, cfg=CFG):
    from scipy.special import expit
    xgb = _require_xgboost()
    p = dict(cfg["XGB"])
    n_est = p.pop("n_estimators")
    params = dict(p, seed=seed, nthread=max(cfg["N_JOBS"], 1),
                  disable_default_eval_metric=1 if use_cost else 0)
    if not use_cost:
        params.update(
            objective="binary:logistic", eval_metric="aucpr",
            scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xval, label=yval)
    kw = dict(num_boost_round=n_est, evals=[(dva, "val")],
              early_stopping_rounds=cfg["EARLY_STOP"], verbose_eval=False)
    if use_cost:
        kw["obj"] = cost_objective(cfg["C_FP"], cfg["C_FN"])
        kw["custom_metric"] = lambda pr, dm: (
            "cost_ec",
            float(np.mean(cfg["C_FN"] * dm.get_label() * (1 - expit(pr))
                          + cfg["C_FP"] * (1 - dm.get_label()) * expit(pr))))
    return xgb.train(params, dtr, **kw)


def best_tau(p_val, y_val, r=None, cfg=CFG):
    """Decision threshold chosen ON THE VAL BLOCK."""
    r = cfg["C_FN"] / cfg["C_FP"] if r is None else r
    pi1 = y_val.mean(); pi0 = 1 - pi1
    best = (0.5, np.inf)
    for t in np.linspace(0, 1, 501):
        pred = p_val >= t
        fp = np.count_nonzero(pred & (y_val == 0))
        tn = np.count_nonzero(~pred & (y_val == 0))
        fn = np.count_nonzero(~pred & (y_val == 1))
        tp = np.count_nonzero(pred & (y_val == 1))
        ec = pi0 * fp / max(fp + tn, 1) + r * pi1 * fn / max(fn + tp, 1)
        if ec < best[1]:
            best = (float(t), float(ec))
    return best


# ============================================================================
#  Probability calibration and out-of-sample ECE
# ============================================================================
def ece(p, y, n_bins=15):
    """Equal-mass expected calibration error."""
    p = np.asarray(p, float); y = np.asarray(y, int)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    tot = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / len(y))


def calibrate(raw_val, y_val, raw_other: dict):
    """Fit Platt and isotonic on val, select by ECE on val, and report the
    out-of-sample ECE on every other block."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    cands = {}
    ir = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val)
    cands["isotonic"] = lambda z: np.clip(ir.predict(z), 1e-6, 1 - 1e-6)
    lr = LogisticRegression(max_iter=1000).fit(raw_val.reshape(-1, 1), y_val)
    cands["platt"] = lambda z: lr.predict_proba(z.reshape(-1, 1))[:, 1]

    name = min(cands, key=lambda k: ece(cands[k](raw_val), y_val))
    f = cands[name]
    out = {k: f(v[0]) for k, v in raw_other.items()}
    ece_out = {k: ece(out[k], raw_other[k][1]) for k in raw_other}
    return name, f(raw_val), out, ece(f(raw_val), y_val), ece_out


# ============================================================================
#  Conformal prediction
# ============================================================================
def nonconformity(p, y):
    """s(x, y) = 1 - p if y = 1, p if y = 0."""
    return np.where(y == 1, 1 - p, p)


def conformal_quantile(scores, level):
    """Split-conformal quantile at the given coverage level."""
    n = len(scores)
    if n == 0:
        return 1.0
    lv = min(np.ceil((n + 1) * level) / n, 1.0)
    return float(np.quantile(scores, lv, method="higher"))


def conformal(p_cal, y_cal, p_test, y_test, alphas=None, cfg=CFG):
    """Marginal split-conformal predictor: one quantile over all of cal."""
    from scipy.stats import ks_2samp
    alphas = cfg["ALPHA_SIG"] if alphas is None else alphas
    s_cal = nonconformity(p_cal, y_cal)
    s_te = nonconformity(p_test, y_test)
    n = len(s_cal)

    ks_marg = float(ks_2samp(s_cal, s_te).statistic)
    ks_frd = (float(ks_2samp(s_cal[y_cal == 1], s_te[y_test == 1]).statistic)
              if (y_cal == 1).any() and (y_test == 1).any() else np.nan)

    rows = []
    for a in alphas:
        q = conformal_quantile(s_cal, 1 - a)
        inc1 = (1 - p_test) <= q
        inc0 = p_test <= q
        covered = np.where(y_test == 1, inc1, inc0)
        m = y_test == 1
        rows.append(dict(
            alpha_sig=a, n_cal=n, n_cal_fraud=int((y_cal == 1).sum()),
            q_hat=round(q, 6),
            cov_marginal=round(float(covered.mean()), 4),
            cov_fraud_class=round(float(covered[m].mean()), 4) if m.any() else np.nan,
            target=round(1 - a, 4),
            marginal_ok=bool(covered.mean() >= 1 - a),
            fraud_ok=bool(m.any() and covered[m].mean() >= 1 - a),
            ks_marginal=round(ks_marg, 4), ks_fraud=round(ks_frd, 4),
            singleton_frac=round(float((inc0 ^ inc1).mean()), 4),
            ambiguous_frac=round(float((inc0 & inc1).mean()), 4),
        ))
    return pd.DataFrame(rows)


def mondrian_conformal(p_cal, y_cal, p_test, y_test, alphas=None, cfg=CFG):
    """Class-conditional (Mondrian) predictor: one quantile per class.

    This is the construction the regulatory reading requires. Marginal
    coverage gives P(y not in Gamma) <= alpha; it does NOT give
    P(1 not in Gamma | y = 1) <= alpha, which needs a quantile computed on the
    calibration frauds alone.
    """
    alphas = cfg["ALPHA_SIG"] if alphas is None else alphas
    s_cal = nonconformity(p_cal, y_cal)
    s_f, s_l = s_cal[y_cal == 1], s_cal[y_cal == 0]
    rows = []
    for a in alphas:
        q_f = conformal_quantile(s_f, 1 - a)
        q_l = conformal_quantile(s_l, 1 - a)
        inc1 = (1 - p_test) <= q_f
        inc0 = p_test <= q_l
        cov = np.where(y_test == 1, inc1, inc0)
        m = y_test == 1
        rows.append(dict(
            alpha_sig=a, n_cal_fraud=int(len(s_f)),
            q_fraud=round(q_f, 6), q_legit=round(q_l, 6),
            cov_fraud_class=round(float(cov[m].mean()), 4) if m.any() else np.nan,
            cov_legit_class=round(float(cov[~m].mean()), 4),
            cov_marginal=round(float(cov.mean()), 4),
            fraud_ok=bool(m.any() and cov[m].mean() >= 1 - a),
            alert_volume_pct=round(100 * float(inc1.mean()), 4),
            ambiguous_frac=round(float((inc0 & inc1).mean()), 4),
        ))
    return pd.DataFrame(rows)


def coverage_price(p_cal, y_cal, p_test, y_test, grid=None):
    """The price of the guarantee: for each target fraud-class coverage, the
    alert volume it costs and the resulting fraud rate among approved
    transactions. This is the source of Table 4 of the paper.
    """
    grid = np.arange(0.50, 1.00, 0.05) if grid is None else grid
    s_f = nonconformity(p_cal, y_cal)[y_cal == 1]
    rows = []
    for target in grid:
        q = conformal_quantile(s_f, float(target))
        flagged = (1 - p_test) <= q
        m = y_test == 1
        approved = ~flagged
        rows.append(dict(
            target_fraud_coverage=round(float(target), 3),
            achieved_fraud_coverage=round(float(flagged[m].mean()), 4) if m.any() else np.nan,
            alert_volume_pct=round(100 * float(flagged.mean()), 4),
            n_alerts=int(flagged.sum()),
            frauds_caught=int(flagged[m].sum()),
            lambda_approved_pct=round(100 * float(y_test[approved].mean()), 6)
                                if approved.any() else np.nan,
        ))
    return pd.DataFrame(rows)


def psd2_indicator_marginal_rule(p_cal, y_cal, p_test, y_test, alpha_sig=0.05):
    """Approved-transaction fraud rate under the MARGINAL conformal rule.

    NOT the quantity reported in the paper. Table 4 and Section 6.4 use the
    class-conditional rule, i.e. `coverage_price(...).lambda_approved_pct`.
    This function is kept only because `all_results.json` records it under the
    key "psd2"; do not compare its output with the paper.
    """
    s_cal = nonconformity(p_cal, y_cal)
    q = conformal_quantile(s_cal, 1 - alpha_sig)
    flagged = (1 - p_test) <= q
    approved = ~flagged
    n_app = int(approved.sum())
    return dict(alpha_sig=alpha_sig, q_hat=round(q, 6), n_approved=n_app,
                frauds_among_approved=int(y_test[approved].sum()) if n_app else 0,
                lambda_pct=round(100 * float(y_test[approved].mean()), 6) if n_app else np.nan,
                alert_volume_pct=round(100 * float(flagged.mean()), 4))


def save_scores(dataset, seed, p_cal, y_cal, p_test, y_test, cfg=CFG):
    """Store the calibrated probabilities so that every later conformal
    analysis takes seconds and no retraining."""
    d = out_dir(cfg) / "scores"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{dataset}_seed{seed}.npz",
                        p_cal=p_cal, y_cal=y_cal, p_test=p_test, y_test=y_test)


# ============================================================================
#  Stage 3 — SemiSync contamination feedback
# ============================================================================
def semisync(S, seed, cfg=CFG, max_iter=None):
    from scipy.special import expit
    xgb = _require_xgboost()
    max_iter = cfg["SS_MAX_ITER"] if max_iter is None else max_iter
    contam, log, prev = cfg["CONTAM_INIT"], [], None
    state = None
    for t in range(max_iter):
        trio = stage0(S, seed, contamination=contam)
        gate = adaptive_gate(trio["val"], S["val"][1], seed)
        a = gate["alpha"]
        Xa = {k: augment(S[k][0], trio[k], a) for k in S}
        bst = train_booster(Xa["train"], S["train"][1],
                            Xa["val"], S["val"][1], seed, use_cost=True)
        raw = {k: expit(bst.predict(xgb.DMatrix(Xa[k]))) for k in S}
        phi = float((raw["train"] > cfg["SS_CONF"]).mean())
        new = float(np.clip(phi, cfg["CONTAM_LO"], cfg["CONTAM_HI"]))
        log.append(dict(iter=t, contamination=contam, phi=round(phi, 6),
                        next_contamination=round(new, 6),
                        alpha=list(np.round(a, 4)),
                        ec_star=round(gate["ec_star"], 6)))
        state = (trio, gate, bst, Xa, raw)
        if prev is not None and abs(phi - prev) < cfg["SS_TOL"]:
            break
        prev, contam = phi, new
    df = pd.DataFrame(log)
    phis = df["phi"].values
    df.attrs["monotone"] = bool(np.all(np.diff(phis) >= -1e-9)
                                or np.all(np.diff(phis) <= 1e-9))
    df.attrs["oscillates"] = not df.attrs["monotone"]
    return df, state


# ============================================================================
#  One dataset, one seed
# ============================================================================
def metrics_at(p, y, tau, cfg=CFG):
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 precision_score, recall_score, f1_score,
                                 matthews_corrcoef)
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return dict(tau=round(float(tau), 4),
                precision=round(precision_score(y, pred, zero_division=0), 4),
                recall=round(recall_score(y, pred, zero_division=0), 4),
                f1=round(f1_score(y, pred, zero_division=0), 4),
                mcc=round(matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0, 4),
                auprc=round(average_precision_score(y, p), 4),
                roc_auc=round(roc_auc_score(y, p), 4),
                TP=tp, FP=fp, FN=fn, TN=tn,
                raw_cost=int(cfg["C_FP"] * fp + cfg["C_FN"] * fn))


def run_one(name, S, seed, cfg=CFG, verbose=True):
    xgb = _require_xgboost()
    t0 = time.time()
    res = dict(dataset=name, seed=seed)
    res["splits"] = {k: dict(n=int(len(S[k][1])), frauds=int(S[k][1].sum()))
                     for k in S}

    ss_log, state = semisync(S, seed, cfg)
    trio, gate, bst, Xa, raw = state
    res["semisync"] = ss_log.to_dict("records")
    res["semisync_monotone"] = ss_log.attrs["monotone"]
    res["gate"] = {k: (list(map(float, v)) if isinstance(v, np.ndarray) else float(v))
                   for k, v in gate.items()}

    yv, yc, yt = S["val"][1], S["cal"][1], S["test"][1]
    models = {}

    # 1) cost-sensitive detector (ADAPTIVE-CP-FRAUD)
    cname, p_val, others, ece_in, ece_out = calibrate(
        raw["val"], yv, {"cal": (raw["cal"], yc), "test": (raw["test"], yt)})
    tau, _ = best_tau(p_val, yv, cfg=cfg)
    models["ADAPTIVE-CP-FRAUD"] = dict(
        **metrics_at(others["test"], yt, tau, cfg), calibrator=cname,
        ece_in_sample_val=round(ece_in, 5),
        ece_out_of_sample_test=round(ece_out["test"], 5))
    p_cal_adp, p_te_adp = others["cal"], others["test"]

    # 2) Baseline SPW — no anomaly features
    b2 = train_booster(S["train"][0], S["train"][1], S["val"][0], yv,
                       seed, use_cost=False, cfg=cfg)
    r2 = {k: b2.predict(xgb.DMatrix(S[k][0])) for k in S}
    _, pv2, o2, _, e2 = calibrate(r2["val"], yv,
                                  {"cal": (r2["cal"], yc), "test": (r2["test"], yt)})
    t2, _ = best_tau(pv2, yv, cfg=cfg)
    models["Baseline SPW"] = dict(**metrics_at(o2["test"], yt, t2, cfg),
                                  ece_out_of_sample_test=round(e2["test"], 5))

    # 3) HybridMeta-XGB (Hybrid-XGB in the paper) — anomalies, uniform weights
    au = np.array([1 / 3, 1 / 3, 1 / 3])
    Xu = {k: augment(S[k][0], trio[k], au) for k in S}
    b3 = train_booster(Xu["train"], S["train"][1], Xu["val"], yv,
                       seed, use_cost=False, cfg=cfg)
    r3 = {k: b3.predict(xgb.DMatrix(Xu[k])) for k in S}
    _, pv3, o3, _, e3 = calibrate(r3["val"], yv,
                                  {"cal": (r3["cal"], yc), "test": (r3["test"], yt)})
    t3, _ = best_tau(pv3, yv, cfg=cfg)
    models["HybridMeta-XGB"] = dict(**metrics_at(o3["test"], yt, t3, cfg),
                                    ece_out_of_sample_test=round(e3["test"], 5))
    res["models"] = models

    # conformal stage — on the calibration block
    res["conformal"] = conformal(p_cal_adp, yc, p_te_adp, yt, cfg=cfg).to_dict("records")
    res["psd2"] = [psd2_indicator_marginal_rule(p_cal_adp, yc, p_te_adp, yt, a)
                   for a in cfg["ALPHA_SIG"]]
    res["mondrian"] = mondrian_conformal(p_cal_adp, yc, p_te_adp, yt, cfg=cfg).to_dict("records")
    res["tradeoff"] = coverage_price(p_cal_adp, yc, p_te_adp, yt).to_dict("records")
    save_scores(name, seed, p_cal_adp, yc, p_te_adp, yt, cfg)

    # ablations (test block, threshold from val)
    abl = []
    for lbl, alpha_used, use_cost in [("full", gate["alpha"], True),
                                      ("no_stage1_uniform", au, True),
                                      ("no_stage2_spw", gate["alpha"], False)]:
        Xz = {k: augment(S[k][0], trio[k], alpha_used) for k in S}
        bz = train_booster(Xz["train"], S["train"][1], Xz["val"], yv,
                           seed, use_cost, cfg=cfg)
        rz = {k: bz.predict(xgb.DMatrix(Xz[k])) for k in S}
        _, pvz, oz, _, _ = calibrate(rz["val"], yv, {"test": (rz["test"], yt)})
        tz, _ = best_tau(pvz, yv, cfg=cfg)
        abl.append(dict(config=lbl, **metrics_at(oz["test"], yt, tz, cfg)))
    base_row = {k: v for k, v in models["Baseline SPW"].items()
                if k != "ece_out_of_sample_test"}
    abl.append(dict(config="baseline_no_anomaly", **base_row))
    res["ablation"] = abl

    res["runtime_min"] = round((time.time() - t0) / 60, 2)
    if verbose:
        print(f"  [{name} seed={seed}] {res['runtime_min']} min | "
              f"alpha*={gate['alpha']} | "
              f"AUPRC={models['ADAPTIVE-CP-FRAUD']['auprc']}")
    return res


def run_dataset(name, X, y, extra_test=None, seeds=None, cfg=CFG, ALL=None,
                verbose=True):
    from sklearn.preprocessing import StandardScaler
    o = _ensure_out(cfg)
    if verbose:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
    S = four_way(X, y, cfg)
    if extra_test is not None:            # Sparkov, provider split
        Xte, yte = extra_test
        # same slice as four_way used, so the same scaler
        sc = StandardScaler().fit(X[:int(len(y) * cfg["FRAC_TRAIN"])])
        S["test"] = (sc.transform(Xte), yte)
    split_df = report_split(name, S, verbose)
    split_df.to_csv(o / "tables" / f"table02_splits_{name}.csv", index=False)
    seeds = cfg["SEEDS"] if seeds is None else seeds
    runs = [run_one(name, S, s, cfg, verbose) for s in seeds]
    if ALL is not None:
        ALL[name] = runs
    return runs


# ============================================================================
#  Aggregation helpers
# ============================================================================
def aggregate(runs):
    rows = []
    for r in runs:
        for m, v in r["models"].items():
            rows.append(dict(dataset=r["dataset"], seed=r["seed"], model=m, **v))
    return pd.DataFrame(rows)


def summarise(df, cols=("auprc", "roc_auc", "f1", "mcc",
                        "precision", "recall", "raw_cost")):
    g = df.groupby(["dataset", "model"])[list(cols)]
    out = g.agg(["mean", "std"]).round(4)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def seed_stability(ALL):
    rows, alpha_rows = [], []
    for ds, runs in ALL.items():
        for r in runs:
            a = r["gate"]["alpha"]
            alpha_rows.append(dict(
                dataset=ds, seed=r["seed"],
                alpha=str([round(float(x), 3) for x in a]),
                vertex=("IF" if a[0] > 0.99 else "LOF" if a[1] > 0.99
                        else "OC" if a[2] > 0.99 else "interior"),
                ec_star=round(float(r["gate"]["ec_star"]), 5),
                gain_pct=round(float(r["gate"]["gain_pct"]), 3)))
        for m in ("ADAPTIVE-CP-FRAUD", "Baseline SPW", "HybridMeta-XGB"):
            v = np.array([r["models"][m]["auprc"] for r in runs], float)
            rows.append(dict(
                dataset=ds, model=m, n_seeds=len(v),
                auprc_mean=round(v.mean(), 4),
                auprc_sd=round(v.std(ddof=1), 4) if len(v) > 1 else np.nan,
                auprc_min=round(v.min(), 4), auprc_max=round(v.max(), 4)))
    return pd.DataFrame(alpha_rows), pd.DataFrame(rows)


# ============================================================================
#  Significance testing — Table B.2 of the paper
# ============================================================================
def holm(pvals, labels=None):
    """Holm step-down adjustment. Returns an array, or a dict if labels given."""
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    if labels is None:
        return adj
    return {labels[i]: round(float(adj[i]), 4) for i in range(m)}


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation for the signed-rank test.

    Positive means the first model of the pair is the more expensive one.
    """
    d = np.asarray(d, float)
    nz = d[d != 0]
    if nz.size == 0:
        return 0.0
    ranks = pd.Series(np.abs(nz)).rank().values
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    tot = w_pos + w_neg
    return float((w_pos - w_neg) / tot) if tot > 0 else 0.0


PAIRS = [("ADAPTIVE-CP-FRAUD", "Baseline SPW", "Cost-sens. vs Baseline"),
         ("ADAPTIVE-CP-FRAUD", "HybridMeta-XGB", "Cost-sens. vs Hybrid"),
         ("Baseline SPW", "HybridMeta-XGB", "Baseline vs Hybrid")]

DATASET_NAMES = {"T": "ULB", "K": "Sparkov", "PS": "PaySim"}


def wilcoxon_holm(models_csv=None, cfg=CFG):
    """Paired Wilcoxon signed-rank on the per-seed raw costs, Holm-corrected
    within each dataset over its three comparisons. Reproduces Table B.2.

    With three PaySim seeds the smallest attainable two-sided p-value is 0.25,
    so no PaySim comparison can reach significance; the rows are reported for
    completeness.
    """
    from scipy.stats import wilcoxon
    o = out_dir(cfg)
    models_csv = (o / "tables" / "paper_models.csv") if models_csv is None else Path(models_csv)
    M = pd.read_csv(models_csv)
    piv = M.pivot_table(index=["dataset", "seed"], columns="model",
                        values="raw_cost")
    out = []
    for ds in [d for d in ("T", "K", "PS") if d in piv.index.get_level_values(0)]:
        sub = piv.loc[ds]
        pvals, rows = [], []
        for a, b, lab in PAIRS:
            d = (sub[a] - sub[b]).values.astype(float)
            try:
                p = float(wilcoxon(sub[a].values, sub[b].values).pvalue)
            except ValueError:            # all differences zero
                p = 1.0
            pvals.append(p)
            rows.append(dict(dataset=DATASET_NAMES.get(ds, ds), comparison=lab,
                             p=round(p, 3), r=round(rank_biserial(d), 2),
                             cheaper=f"{int((d < 0).sum())}/{len(d)}"))
        adj = holm(pvals)
        for row, pa in zip(rows, adj):
            row["p_holm"] = round(float(pa), 3)
            out.append(row)
    return pd.DataFrame(out)[["dataset", "comparison", "p", "p_holm",
                              "r", "cheaper"]]


# ============================================================================
#  Reproduction from the released score vectors
# ============================================================================
def load_scores(scores_dir=None, cfg=CFG):
    """{dataset: {seed: (p_cal, y_cal, p_test, y_test)}} from results/scores."""
    d = Path(scores_dir) if scores_dir else out_dir(cfg) / "scores"
    out = {}
    for f in sorted(glob.glob(str(d / "*.npz"))):
        ds, sd = Path(f).stem.split("_seed")
        z = np.load(f)
        out.setdefault(ds, {})[int(sd)] = (z["p_cal"], z["y_cal"],
                                           z["p_test"], z["y_test"])
    if not out:
        raise FileNotFoundError(
            f"no .npz found in {d}. The released score vectors ship in "
            f"results/scores/; set $CFRAUD_OUT_DIR if they live elsewhere.")
    return out


def _ordered(S):
    return ([d for d in ("T", "K", "PS") if d in S]
            + [d for d in S if d not in ("T", "K", "PS")])


def table_coverage(S, cfg=CFG):
    """Per-seed marginal against class-conditional coverage (Table 3)."""
    from scipy.stats import ks_2samp
    rows = []
    for ds in _ordered(S):
        for sd, (pc, yc, pt, yt) in S[ds].items():
            sc, st = nonconformity(pc, yc), nonconformity(pt, yt)
            m = yt == 1
            ks_marg = float(ks_2samp(sc, st).statistic)
            ks_frd = float(ks_2samp(sc[yc == 1], st[m]).statistic) if m.any() else np.nan
            for a in cfg["ALPHA_SIG"]:
                qm = conformal_quantile(sc, 1 - a)
                cov_m = np.where(m, (1 - pt) <= qm, pt <= qm)
                qf = conformal_quantile(sc[yc == 1], 1 - a)
                ql = conformal_quantile(sc[yc == 0], 1 - a)
                inc1, inc0 = (1 - pt) <= qf, pt <= ql
                cov_c = np.where(m, inc1, inc0)
                rows.append(dict(
                    dataset=ds, seed=sd, alpha_sig=a, target=round(1 - a, 3),
                    marg_cov_marginal=round(float(cov_m.mean()), 4),
                    marg_cov_fraud=round(float(cov_m[m].mean()), 4),
                    marg_fraud_ok=bool(cov_m[m].mean() >= 1 - a),
                    mond_cov_marginal=round(float(cov_c.mean()), 4),
                    mond_cov_fraud=round(float(cov_c[m].mean()), 4),
                    mond_fraud_ok=bool(cov_c[m].mean() >= 1 - a),
                    mond_alert_pct=round(100 * float(inc1.mean()), 4),
                    n_cal_fraud=int((yc == 1).sum()),
                    ks_marginal=round(ks_marg, 4), ks_fraud=round(ks_frd, 4)))
    return pd.DataFrame(rows)


def table_price(S):
    """Per-seed price of the class-conditional guarantee (Table 4)."""
    rows = []
    for ds in _ordered(S):
        for sd, (pc, yc, pt, yt) in S[ds].items():
            df = coverage_price(pc, yc, pt, yt)
            df.insert(0, "seed", sd)
            df.insert(0, "dataset", ds)
            df["base_rate_pct"] = round(100 * float(yt.mean()), 6)
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def table_gap(S, cfg=CFG):
    """Per-seed coverage dissociation used by Figure 2."""
    rows = []
    for ds in _ordered(S):
        for sd, (pc, yc, pt, yt) in S[ds].items():
            sc, m = nonconformity(pc, yc), (yt == 1)
            for a in cfg["ALPHA_SIG"]:
                qm = conformal_quantile(sc, 1 - a)
                cov_m = np.where(m, (1 - pt) <= qm, pt <= qm)
                qf = conformal_quantile(sc[yc == 1], 1 - a)
                ql = conformal_quantile(sc[yc == 0], 1 - a)
                cov_c = np.where(m, (1 - pt) <= qf, pt <= ql)
                rows.append(dict(dataset=ds, seed=sd, alpha=a,
                                 marg_all=cov_m.mean(),
                                 marg_fraud=cov_m[m].mean(),
                                 cond_fraud=cov_c[m].mean()))
    return pd.DataFrame(rows)


NCAL_GRID = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 250, 500, 1000]
NCAL_RESAMPLES = 40
NCAL_TARGETS = [0.80, 0.90, 0.95]


def table_ncal_sweep(S, grid=None, n_resample=NCAL_RESAMPLES,
                     targets=NCAL_TARGETS):
    """Alert volume and degeneracy against the calibration fraud count.

    The test scores are sorted once and queried by binary search, which is what
    makes the PaySim sweep take seconds rather than hours.
    """
    grid = NCAL_GRID if grid is None else grid
    rows = []
    for ds in _ordered(S):
        for sd, (pc, yc, pt, yt) in S[ds].items():
            s_test_all = np.sort(1.0 - pt)
            s_test_fraud = np.sort((1.0 - pt)[yt == 1])
            n_te, n_fr = len(s_test_all), len(s_test_fraud)
            s_f_all = np.sort(1.0 - pc[yc == 1])
            n_all = len(s_f_all)
            g = np.random.default_rng(10_000 + sd)

            def counts(q):
                return (np.searchsorted(s_test_all, q, side="right") / n_te,
                        np.searchsorted(s_test_fraud, q, side="right") / max(n_fr, 1))

            sizes = [n for n in grid if n < n_all] + [n_all]
            for n_f in sizes:
                for rep in range(1 if n_f == n_all else n_resample):
                    s_f = (s_f_all if n_f == n_all
                           else np.sort(s_f_all[g.choice(n_all, n_f, replace=False)]))
                    for t in targets:
                        q = conformal_quantile(s_f, t)
                        av, ach = counts(q)
                        rows.append(dict(dataset=ds, seed=sd, n_cal_fraud=n_f,
                                         resample=rep, target=t,
                                         achieved=round(float(ach), 4),
                                         alert_pct=round(100 * float(av), 4),
                                         degenerate=bool(av > 0.5)))
    return pd.DataFrame(rows)


def summarise_ncal(D):
    return (D.groupby(["dataset", "target", "n_cal_fraud"])
             .agg(alert_median=("alert_pct", "median"),
                  alert_p10=("alert_pct", lambda x: np.percentile(x, 10)),
                  alert_p90=("alert_pct", lambda x: np.percentile(x, 90)),
                  achieved_mean=("achieved", "mean"),
                  p_degenerate=("degenerate", "mean"))
             .round(4).reset_index())


# ============================================================================
#  Figures
#
#  Titles are deliberately NOT drawn on the images: Elsevier asks that a
#  figure's title live in its caption, not on the artwork. Datasets are named,
#  never abbreviated to the pipeline's internal T / K / PS codes.
# ============================================================================
FIG_DPI = 400


def _fmt_pct(v):
    """Two significant figures, but never scientific notation (100, not 1e+02)."""
    s = f"{v:.2g}"
    return f"{v:.0f}" if "e" in s else s


def make_figures(S, price_df=None, ncal_summary=None, fig_dir=None,
                 dpi=FIG_DPI, cfg=CFG):
    """Regenerate the paper's four data figures. Returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir) if fig_dir else (REPO_ROOT / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = _ordered(S)
    written = []

    # ---- Figure 2 of the paper: the coverage dissociation -------------------
    G = table_gap(S, cfg)
    alphas = sorted(G["alpha"].unique())
    g = G.groupby(["dataset", "alpha"]).agg(["mean", "std"])
    fig, axes = plt.subplots(1, len(order), figsize=(4.6 * len(order), 4.0),
                             sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), order):
        x = np.arange(len(alphas)); w = 0.27
        sub = g.loc[ds]
        for k, (col, lab, c) in enumerate([
                ("marg_all", "marginal predictor: overall coverage", "#4C78A8"),
                ("marg_fraud", "marginal predictor: fraud-class coverage", "#E45756"),
                ("cond_fraud", "class-conditional: fraud-class coverage", "#59A14F")]):
            ax.bar(x + (k - 1) * w, sub[(col, "mean")].values, w, label=lab,
                   color=c, yerr=sub[(col, "std")].values, capsize=2,
                   error_kw=dict(lw=.8))
        for i, a in enumerate(alphas):
            ax.hlines(1 - a, i - 1.6 * w, i + 1.6 * w, ls="--", lw=1.2,
                      color="black", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\alpha={a}$" for a in alphas])
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_NAMES.get(ds, ds))
        ax.grid(axis="y", alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("Empirical coverage")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="center left", framealpha=.95)
    fig.tight_layout()
    p = fig_dir / "fig_coverage_gap.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    # ---- Figure 3 of the paper: the mechanism, on one seed ------------------
    targets = [0.75, 0.90, 0.95]
    fig, axes = plt.subplots(1, len(order), figsize=(4.8 * len(order), 4.2),
                             sharey=True)
    for ax, ds in zip(np.atleast_1d(axes), order):
        sd = sorted(S[ds])[0]
        pc, yc, pt, yt = S[ds][sd]
        s_f = np.sort(nonconformity(pc, yc)[yc == 1])
        s_lg = np.sort((1 - pt)[yt == 0])
        ax.plot(s_f, np.arange(1, len(s_f) + 1) / len(s_f), lw=2,
                color="#E45756", label=f"calibration frauds ($n={len(s_f)}$)")
        ax.plot(s_lg, np.arange(1, len(s_lg) + 1) / len(s_lg), lw=2,
                color="#4C78A8", label="test legitimate traffic")
        for t, col in zip(targets, ["#999999", "#666666", "#111111"]):
            q = conformal_quantile(s_f, t)
            av = float(((1 - pt) <= q).mean())
            ax.vlines(q, 0, t, ls=":", lw=1.3, color=col)
            ax.hlines(t, 0, q, ls=":", lw=1.3, color=col)
            ax.annotate(f"{t:.0%} → {_fmt_pct(100 * av)}% alerts", xy=(q, t),
                        xytext=(6, -11), textcoords="offset points",
                        fontsize=8, color=col)
        ax.set_xlabel("Nonconformity score  $s = 1-\\hat{p}$")
        ax.set_title(DATASET_NAMES.get(ds, ds))
        ax.grid(alpha=.3); ax.set_xlim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("Empirical CDF")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(w_pad=3.2)
    p = fig_dir / "fig_mechanism.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig); written.append(p)

    # ---- Figure 4 of the paper: the price curve -----------------------------
    C = table_price(S) if price_df is None else price_df
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ds, sub in C.groupby("dataset"):
        gg = sub.groupby("target_fraud_coverage")["alert_volume_pct"]
        mu, sd = gg.mean(), gg.std()
        ax.plot(mu.index, mu.values, marker="o", label=DATASET_NAMES.get(ds, ds))
        ax.fill_between(mu.index, mu - sd, mu + sd, alpha=0.18)
    ax.set_xlabel("Target fraud-class coverage $1-\\alpha$")
    ax.set_ylabel("Alert volume (% of transactions)")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    p = fig_dir / "fig_coverage_price.png"
    fig.savefig(p, dpi=dpi); plt.close(fig); written.append(p)

    # ---- Figure 5 of the paper: price against calibration fraud count -------
    summ = ncal_summary if ncal_summary is not None else summarise_ncal(table_ncal_sweep(S))
    fig, axes = plt.subplots(1, len(NCAL_TARGETS), figsize=(13, 4), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), NCAL_TARGETS):
        for ds in order:
            s = summ[(summ.target == t) & (summ.dataset == ds)].sort_values("n_cal_fraud")
            ax.plot(s.n_cal_fraud, s.alert_median, marker="o",
                    label=DATASET_NAMES.get(ds, ds))
            ax.fill_between(s.n_cal_fraud, s.alert_p10, s.alert_p90, alpha=.15)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Frauds in calibration split")
        ax.set_title(f"Target fraud coverage {t:.2f}")
        ax.grid(alpha=.3, which="both")
    np.atleast_1d(axes)[0].set_ylabel("Alert volume (%)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    p = fig_dir / "fig_ncal_price.png"
    fig.savefig(p, dpi=dpi); plt.close(fig); written.append(p)

    return written


# ============================================================================
#  Entry points
# ============================================================================
def reproduce(cfg=CFG, make_figs=True, verbose=True):
    """Recompute every conformal table and figure from results/scores/*.npz."""
    o = _ensure_out(cfg)
    S = load_scores(cfg=cfg)
    if verbose:
        print("score vectors:", {DATASET_NAMES.get(k, k): sorted(v)
                                 for k, v in S.items()})

    A = table_coverage(S, cfg); A.to_csv(o / "tables" / "paper_A_coverage.csv", index=False)
    C = table_price(S);         C.to_csv(o / "tables" / "paper_C_price.csv", index=False)
    E = table_gap(S, cfg);      E.to_csv(o / "tables" / "paper_E_gap.csv", index=False)
    D = table_ncal_sweep(S);    D.to_csv(o / "tables" / "paper_D_ncal_sweep.csv", index=False)
    summ = summarise_ncal(D)

    if verbose:
        print("\n--- Table 3: marginal against class-conditional coverage ---")
        print(A.groupby(["dataset", "alpha_sig"])[
            ["marg_cov_marginal", "marg_cov_fraud",
             "mond_cov_fraud", "mond_alert_pct"]].mean().round(4).to_string())
        print("\n--- Table 4: the price of the guarantee ---")
        print(C.groupby(["dataset", "target_fraud_coverage"])[
            ["achieved_fraud_coverage", "alert_volume_pct",
             "lambda_approved_pct"]].mean().round(4).to_string())

    models_csv = o / "tables" / "paper_models.csv"
    if models_csv.exists():
        W = wilcoxon_holm(models_csv, cfg)
        W.to_csv(o / "tables" / "paper_wilcoxon.csv", index=False)
        if verbose:
            print("\n--- Table B.2: paired Wilcoxon, Holm-corrected ---")
            print(W.to_string(index=False))
    elif verbose:
        print(f"\n(paper_models.csv absent, skipping Table B.2)")

    if make_figs:
        paths = make_figures(S, price_df=C, ncal_summary=summ, cfg=cfg)
        if verbose:
            print("\nfigures written:")
            for p in paths:
                print("  ", p)
    return dict(coverage=A, price=C, gap=E, ncal=D)


def train(cfg=CFG, datasets=("T", "K", "PS"), verbose=True):
    """Re-run the full experiment from the raw CSVs. Needs the Kaggle files."""
    o = _ensure_out(cfg)
    ALL = {}
    missing = check_data(cfg)
    if missing and ("T" in datasets or "K" in datasets):
        raise FileNotFoundError(
            f"missing raw files in {cfg['DATA_DIR']}: {missing}. "
            f"Set $CFRAUD_DATA_DIR, or see data/README.md for the downloads.")

    if "T" in datasets:
        XT, yT = load_T(data_path("FILE_T", cfg))
        run_dataset("T", XT, yT, cfg=cfg, ALL=ALL, verbose=verbose)
    if "K" in datasets:
        XK, yK, extra = load_K(data_path("FILE_K_TRAIN", cfg),
                               data_path("FILE_K_TEST", cfg))
        run_dataset("K", XK, yK, extra_test=extra, cfg=cfg, ALL=ALL,
                    verbose=verbose)
    if "PS" in datasets:
        ps = data_path("FILE_PS", cfg)
        if not ps.exists():
            raise FileNotFoundError(f"missing {ps}; see data/README.md")
        XPS, yPS = load_PS(ps)
        run_dataset("PS", XPS, yPS, seeds=cfg["SEEDS_PS"], cfg=cfg, ALL=ALL,
                    verbose=verbose)

    full = pd.concat([aggregate(v) for v in ALL.values()], ignore_index=True)
    full.to_csv(o / "tables" / "paper_models.csv", index=False)
    abl = pd.concat([pd.DataFrame(r["ablation"]).assign(dataset=r["dataset"],
                                                        seed=r["seed"])
                     for v in ALL.values() for r in v], ignore_index=True)
    abl.to_csv(o / "tables" / "paper_ablation.csv", index=False)
    A, _ = seed_stability(ALL)
    A.to_csv(o / "tables" / "paper_gate.csv", index=False)
    with open(o / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(ALL, f, indent=2, default=str)
    if verbose:
        print(f"\nwritten {o / 'all_results.json'}")
    return ALL


USAGE = """usage: python src/pipeline.py [reproduce|train|selftest]

  reproduce   recompute every conformal table and figure of the paper from
              results/scores/*.npz. No dataset, no xgboost, seconds. (default)
  train       re-run the full experiment from the raw CSVs. Needs the four
              Kaggle files, xgboost, and days of compute.
  selftest    check that the deterministic generators are order-independent.

Environment:
  CFRAUD_DATA_DIR   directory holding the raw CSVs   (default <repo>/data)
  CFRAUD_OUT_DIR    directory for outputs            (default <repo>/results)
"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "reproduce"
    if cmd in ("-h", "--help", "help"):
        print(USAGE); return 0
    if cmd == "selftest":
        ok = selftest_determinism()
        print("deterministic draws, independent of execution order:", ok)
        return 0 if ok else 1
    if cmd == "reproduce":
        reproduce()
        return 0
    if cmd == "train":
        train()
        return 0
    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
