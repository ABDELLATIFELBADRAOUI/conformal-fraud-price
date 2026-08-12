# ============================================================================
#  CELLULE 1 — Configuration
# ============================================================================
import os, json, time, math, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

CFG = dict(
    # --- chemins (les quatre fichiers sont dans le même dossier) -------------
    DATA_DIR     = r"C:\Users\LENOVO\Desktop\doctorat\HYBRID XGB CREDIT CARD\Credit Card Transactions Fraud Detection Dataset",
    FILE_T       = "creditcard.csv",          # ULB
    FILE_K_TRAIN = "fraudTrain.csv",          # Sparkov
    FILE_K_TEST  = "fraudTest.csv",           # Sparkov
    # FILE_V     = "train_transaction.csv",   # IEEE-CIS — ÉCARTÉ (voir cellule 10)
    OUT_DIR      = "revision_outputs_v10",

    # --- C1 : découpage en quatre blocs --------------------------------------
    #   train  : ajustement des détecteurs et du booster
    #   val    : gate alpha*, early stopping, isotonic, seuil tau*
    #   cal    : UNIQUEMENT le quantile conforme
    #   test   : évaluation finale
    FRAC_TRAIN = 0.55, FRAC_VAL = 0.15, FRAC_CAL = 0.15,   # test = le reste

    # --- coûts ---------------------------------------------------------------
    C_FP = 1.0, C_FN = 10.0,
    R_GRID = [2, 5, 10, 20, 50],

    # --- détecteurs ----------------------------------------------------------
    IF_TREES = 200, LOF_K = 20, OCSVM_NU = 0.002,
    LOF_CAP = 50_000, OCSVM_CAP = 20_000,
    CONTAM_INIT = 0.002, CONTAM_LO = 0.001, CONTAM_HI = 0.10,

    # --- XGBoost -------------------------------------------------------------
    XGB = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
               min_child_weight=5),
    EARLY_STOP = 50,

    # --- gate ----------------------------------------------------------------
    NM_STARTS = 10, NM_MAXEVAL = 1000, N_TAU_GRID = 301,

    # --- SemiSync ------------------------------------------------------------
    SS_MAX_ITER = 10, SS_TOL = 1e-3, SS_CONF = 0.95,

    # --- conforme ------------------------------------------------------------
    ALPHA_SIG = [0.05, 0.10, 0.20],

    # --- C5 : reproductibilité ----------------------------------------------
    # IEEE-CIS écarté (~28 h par graine) : le budget de calcul va au multi-graines
    # sur ULB et Sparkov. 10 graines donnent un écart-type utilisable et un test
    # de stabilité de alpha* que le manuscrit n'a jamais fait.
    SEEDS = list(range(42, 52)),    # mettre [42] pour un essai rapide
    N_JOBS = 1,     # 1 = déterministe au bit près. -1 accélère mais la somme
                    # des réductions flottantes multi-thread n'est plus fixée.
    BOOTSTRAP_REPS = 1000,
)

# chemins complets + vérification d'existence, avant toute exécution longue
_D = Path(CFG["DATA_DIR"])
CFG["PATH_T"]       = str(_D / CFG["FILE_T"])
CFG["PATH_K_TRAIN"] = str(_D / CFG["FILE_K_TRAIN"])
CFG["PATH_K_TEST"]  = str(_D / CFG["FILE_K_TEST"])
# CFG["PATH_V"] = str(_D / CFG["FILE_V"])   # IEEE-CIS écarté

print(f"dossier de données : {_D}")
_missing = []
for k in ("PATH_T", "PATH_K_TRAIN", "PATH_K_TEST"):
    ok = os.path.exists(CFG[k])
    sz = f"{os.path.getsize(CFG[k])/1e6:7.1f} Mo" if ok else "  ABSENT"
    print(f"  {'OK ' if ok else '!! '} {sz}  {os.path.basename(CFG[k])}")
    if not ok: _missing.append(os.path.basename(CFG[k]))
if _missing:
    print(f"\n  -> introuvable(s) : {_missing}")
    print("     corriger DATA_DIR ci-dessus, ou n'exécuter que les jeux disponibles "
          "en cellule 10.")

OUT = Path(CFG["OUT_DIR"]); (OUT/"tables").mkdir(parents=True, exist_ok=True)
(OUT/"figures").mkdir(parents=True, exist_ok=True)

import numpy as np, pandas as pd
print("numpy", np.__version__, "| pandas", pd.__version__)
try:
    import xgboost as xgb; print("xgboost", xgb.__version__)
except ImportError:
    raise SystemExit("pip install xgboost")
print("sorties ->", OUT.resolve())

# ============================================================================
#  CELLULE 2 — Générateurs déterministes (C4)
# ============================================================================
import hashlib

def rng_for(seed, name):
    """Generator reproductible, indépendant de l'ordre d'exécution."""
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))

def subsample_idx(n, cap, seed, name):
    """Indices de sous-échantillonnage stables (sans remise)."""
    if n <= cap:
        return np.arange(n)
    return np.sort(rng_for(seed, name).choice(n, cap, replace=False))

# vérification : deux appels donnent le même résultat, dans n'importe quel ordre
_a = subsample_idx(100_000, 20_000, 42, "ocsvm")
_ = rng_for(42, "autre_chose").random(10_000)          # bruit intercalé
_b = subsample_idx(100_000, 20_000, 42, "ocsvm")
assert np.array_equal(_a, _b), "non déterministe"
print("✓ tirages déterministes et insensibles à l'ordre d'exécution")

# ============================================================================
#  CELLULE 3 — Données et découpage 55/15/15/15 (C1)
# ============================================================================
from sklearn.preprocessing import StandardScaler

K_MODE = "provider"     # "provider" = fraudTrain/fraudTest (ce que dit le §3.2)
                        # "single"   = fraudTrain seul, re-découpé

def four_way(X, y, cfg=CFG):
    """Découpage chronologique en train / val / cal / test."""
    n  = len(y)
    c1 = int(n * cfg["FRAC_TRAIN"])
    c2 = c1 + int(n * cfg["FRAC_VAL"])
    c3 = c2 + int(n * cfg["FRAC_CAL"])
    sl = dict(train=slice(0, c1), val=slice(c1, c2),
              cal=slice(c2, c3),  test=slice(c3, n))
    out = {k: (X[s], y[s]) for k, s in sl.items()}
    sc  = StandardScaler().fit(out["train"][0])          # ajusté sur train seul
    return {k: (sc.transform(Xk), yk) for k, (Xk, yk) in out.items()}

def report_split(name, S):
    rows = []
    for k in ("train", "val", "cal", "test"):
        Xk, yk = S[k]
        rows.append(dict(dataset=name, split=k, n=len(yk),
                         frauds=int(yk.sum()),
                         rate_pct=round(100*yk.mean(), 4)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df

def load_T(path):
    df = pd.read_csv(path)
    df = df.sort_values("Time").reset_index(drop=True)
    X = df.drop(columns=["Class", "Amount", "Time"]).copy()
    X["log_amount"] = np.log1p(df["Amount"].values)
    X["time_norm"]  = df["Time"].values / df["Time"].max()
    return X.values.astype(np.float32), df["Class"].values.astype(int)

def load_K(path_train, path_test, mode=K_MODE):
    NUM = ["amt","lat","long","merch_lat","merch_long","city_pop","unix_time","zip"]
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
        return d[cols].fillna(0).values.astype(np.float32), \
               d[lbl].astype(int).values, cat_map
    tr = pd.read_csv(path_train)
    tcol = "unix_time" if "unix_time" in tr.columns else "trans_date_trans_time"
    tr = tr.sort_values(tcol).reset_index(drop=True)
    Xtr, ytr, cmap = feats(tr)
    if mode == "single":
        return Xtr, ytr, None
    te = pd.read_csv(path_test).sort_values(tcol).reset_index(drop=True)
    Xte, yte, _ = feats(te, cmap)
    return Xtr, ytr, (Xte, yte)

def load_V(path):
    df = pd.read_csv(path)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    y  = df["isFraud"].astype(int).values
    drop = ["isFraud","TransactionID","TransactionDT","ProductCD",
            "card4","card6","P_emaildomain","R_emaildomain"] + \
           [f"M{i}" for i in range(1, 10)]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    print(f"  V : {X.shape[1]} colonnes numériques retenues")
    return X.values.astype(np.float32), y

print("✓ chargeurs définis")

# ============================================================================
#  CELLULE 4 — Étage 0
# ============================================================================
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import QuantileTransformer

def stage0(S, seed, contamination=None, cfg=CFG):
    """Retourne {split: (a_if, a_lof, a_oc)}, scores rang-normalisés dans [0,1]."""
    cont = cfg["CONTAM_INIT"] if contamination is None else contamination
    Xtr, ytr = S["train"]
    X0 = Xtr[ytr == 0]                                   # légitimes seulement

    iso = IsolationForest(n_estimators=cfg["IF_TREES"], contamination=cont,
                          random_state=seed, n_jobs=cfg["N_JOBS"]).fit(X0)

    i_lof = subsample_idx(len(X0), cfg["LOF_CAP"], seed, "lof")
    lof = LocalOutlierFactor(n_neighbors=cfg["LOF_K"], novelty=True,
                             n_jobs=cfg["N_JOBS"]).fit(X0[i_lof])

    i_oc = subsample_idx(len(X0), cfg["OCSVM_CAP"], seed, "ocsvm")
    oc  = OneClassSVM(kernel="rbf", gamma="scale",
                      nu=cfg["OCSVM_NU"]).fit(X0[i_oc])

    raw = {k: (-iso.score_samples(S[k][0]),
               -lof.score_samples(S[k][0]),
               -oc .score_samples(S[k][0])) for k in S}

    out = {k: [] for k in S}
    for j in range(3):                                   # QT ajusté sur train
        qt = QuantileTransformer(output_distribution="uniform",
                                 n_quantiles=min(1000, len(S["train"][1])),
                                 random_state=seed)
        qt.fit(raw["train"][j].reshape(-1, 1))
        for k in S:
            out[k].append(qt.transform(raw[k][j].reshape(-1, 1)).ravel())
    return {k: tuple(v) for k, v in out.items()}

print("✓ étage 0 défini")

# ============================================================================
#  CELLULE 5 — Étages 1 et 2
# ============================================================================
from scipy.optimize import minimize
from scipy.special import expit

def expected_cost(scores, y, cfp=None, cfn=None, n_tau=None, cfg=CFG):
    """EC intégré sur la grille de seuils — Définition 5.2 (échelle ~0.6)."""
    cfp = cfg["C_FP"] if cfp is None else cfp
    cfn = cfg["C_FN"] if cfn is None else cfn
    n_tau = cfg["N_TAU_GRID"] if n_tau is None else n_tau
    pi1 = y.mean(); pi0 = 1 - pi1
    n0  = max((y == 0).sum(), 1); n1 = max((y == 1).sum(), 1)
    tot = 0.0
    for t in np.linspace(0, 1, n_tau):
        pred = scores >= t
        fp = np.count_nonzero(pred & (y == 0))
        fn = np.count_nonzero(~pred & (y == 1))
        tot += cfp * (fp / n0) * pi0 + cfn * (fn / n1) * pi1
    return float(tot / n_tau)

def blend(alpha, trio):
    a = np.clip(alpha, 0, None); a = a / (a.sum() + 1e-12)
    return a[0]*trio[0] + a[1]*trio[1] + a[2]*trio[2]

def adaptive_gate(trio_val, y_val, seed, cfg=CFG):
    """Étage 1 : alpha* minimisant l'EC intégré SUR LE BLOC VAL (jamais cal)."""
    obj = lambda al: expected_cost(blend(al, trio_val), y_val)
    best, best_ec, ecs = None, np.inf, []
    starts = [np.array([1/3, 1/3, 1/3])]
    g = rng_for(seed, "nelder_mead")
    starts += [g.dirichlet([1, 1, 1]) for _ in range(cfg["NM_STARTS"] - 1)]
    for s0 in starts:
        r = minimize(obj, s0, method="Nelder-Mead",
                     options=dict(maxfev=cfg["NM_MAXEVAL"], xatol=1e-4, fatol=1e-6))
        a = np.clip(r.x, 0, None); a = a / (a.sum() + 1e-12)
        ec = obj(a); ecs.append(ec)
        if ec < best_ec: best, best_ec = a, ec
    ec_unif = obj(np.array([1/3, 1/3, 1/3]))
    return dict(alpha=np.round(best, 6), ec_star=best_ec, ec_uniform=ec_unif,
                gain_pct=100*(ec_unif - best_ec)/ec_unif,
                ec_var_across_starts=float(np.var(ecs)))

def cost_objective(cfp, cfn):
    """Étage 2 : gradient exact + hessien de substitution défini positif."""
    def obj(pred, dtrain):
        y = dtrain.get_label(); p = expit(pred)
        w = p * (1 - p)
        grad = w * (cfp * (1 - y) - cfn * y)
        surrogate_hessian = w * (cfp * (1 - y) + cfn * y) + 1e-6
        return grad, surrogate_hessian
    return obj

def augment(X, trio, alpha):
    return np.column_stack([X, trio[0], trio[1], trio[2], blend(alpha, trio)])

def train_booster(Xtr, ytr, Xval, yval, seed, use_cost=True, cfg=CFG):
    p = dict(cfg["XGB"]); n_est = p.pop("n_estimators")
    params = dict(p, seed=seed, nthread=max(cfg["N_JOBS"], 1),
                  disable_default_eval_metric=1 if use_cost else 0)
    if not use_cost:
        params.update(objective="binary:logistic", eval_metric="aucpr",
                      scale_pos_weight=float((ytr == 0).sum()/max((ytr == 1).sum(), 1)))
    dtr = xgb.DMatrix(Xtr, label=ytr); dva = xgb.DMatrix(Xval, label=yval)
    kw = dict(num_boost_round=n_est, evals=[(dva, "val")],
              early_stopping_rounds=cfg["EARLY_STOP"], verbose_eval=False)
    if use_cost:
        kw["obj"] = cost_objective(cfg["C_FP"], cfg["C_FN"])
        kw["custom_metric"] = lambda pr, dm: ("cost_ec",
            float(np.mean(cfg["C_FN"]*dm.get_label()*(1-expit(pr)) +
                          cfg["C_FP"]*(1-dm.get_label())*expit(pr))))
    return xgb.train(params, dtr, **kw)

def best_tau(p_val, y_val, r=None, cfg=CFG):
    """C5 : seuil choisi SUR VAL. EC au seuil (échelle des Tableaux 11-15)."""
    r = cfg["C_FN"]/cfg["C_FP"] if r is None else r
    pi1 = y_val.mean(); pi0 = 1 - pi1
    best = (0.5, np.inf)
    for t in np.linspace(0, 1, 501):
        pred = p_val >= t
        fp = np.count_nonzero(pred & (y_val == 0)); tn = np.count_nonzero(~pred & (y_val == 0))
        fn = np.count_nonzero(~pred & (y_val == 1)); tp = np.count_nonzero(pred & (y_val == 1))
        ec = pi0*fp/max(fp+tn, 1) + r*pi1*fn/max(fn+tp, 1)
        if ec < best[1]: best = (float(t), float(ec))
    return best

print("✓ étages 1 et 2 définis")

# ============================================================================
#  CELLULE 6 — Calibration et ECE hors échantillon (C3)
# ============================================================================
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def ece(p, y, n_bins=15):
    """ECE par quantiles."""
    p = np.asarray(p, float); y = np.asarray(y, int)
    edges = np.quantile(p, np.linspace(0, 1, n_bins+1)); edges[0], edges[-1] = -1e-9, 1+1e-9
    tot = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i+1])
        if m.sum() == 0: continue
        tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / len(y))

def calibrate(raw_val, y_val, raw_other: dict):
    """Ajuste Platt et isotonic sur val ; choisit par ECE sur val ;
       renvoie aussi l'ECE hors échantillon sur chaque autre bloc."""
    cands = {}
    ir = IsotonicRegression(out_of_bounds="clip").fit(raw_val, y_val)
    cands["isotonic"] = lambda z: np.clip(ir.predict(z), 1e-6, 1-1e-6)
    lr = LogisticRegression(max_iter=1000).fit(raw_val.reshape(-1, 1), y_val)
    cands["platt"] = lambda z: lr.predict_proba(z.reshape(-1, 1))[:, 1]

    # sélection sur VAL uniquement (jamais sur test)
    name = min(cands, key=lambda k: ece(cands[k](raw_val), y_val))
    f = cands[name]
    out = {k: f(v[0]) for k, v in raw_other.items()}
    ece_out = {k: ece(out[k], raw_other[k][1]) for k in raw_other}   # HORS échantillon
    return name, f(raw_val), out, ece(f(raw_val), y_val), ece_out

print("✓ calibration définie (C3)")

# ============================================================================
#  CELLULE 7 — Étage 4 : conforme (C1) + KS correct (C2)
# ============================================================================
from scipy.stats import ks_2samp

def nonconformity(p, y):
    """s(x,y) = 1-p si y=1, p si y=0 — Définition 5.11."""
    return np.where(y == 1, 1 - p, p)

def conformal(p_cal, y_cal, p_test, y_test, alphas=None, cfg=CFG):
    alphas = cfg["ALPHA_SIG"] if alphas is None else alphas
    s_cal = nonconformity(p_cal, y_cal)          # C1 : bloc cal, intouché
    s_te  = nonconformity(p_test, y_test)
    n = len(s_cal)

    # C2 : populations comparables des deux côtés
    ks_marg = float(ks_2samp(s_cal, s_te).statistic)
    ks_frd  = float(ks_2samp(s_cal[y_cal == 1], s_te[y_test == 1]).statistic) \
              if (y_cal == 1).any() and (y_test == 1).any() else np.nan

    rows = []
    for a in alphas:
        lvl = min(np.ceil((n + 1) * (1 - a)) / n, 1.0)
        q   = float(np.quantile(s_cal, lvl, method="higher"))
        inc1 = (1 - p_test) <= q
        inc0 = p_test <= q
        covered = np.where(y_test == 1, inc1, inc0)
        m = y_test == 1
        rows.append(dict(
            alpha_sig=a, n_cal=n, n_cal_fraud=int((y_cal == 1).sum()), q_hat=round(q, 6),
            cov_marginal=round(float(covered.mean()), 4),
            cov_fraud_class=round(float(covered[m].mean()), 4) if m.any() else np.nan,
            target=round(1 - a, 4),
            marginal_ok=bool(covered.mean() >= 1 - a),
            fraud_ok=bool(m.any() and covered[m].mean() >= 1 - a),
            ks_marginal=round(ks_marg, 4), ks_fraud=round(ks_frd, 4),
            infl_bound_marginal=round(min(a + 2*ks_marg, 1.0), 4),
            infl_bound_fraud=round(min(a + 2*ks_frd, 1.0), 4),
            infl_is_vacuous=bool(min(a + 2*ks_marg, 1.0) >= 0.5),
            singleton_frac=round(float((inc0 ^ inc1).mean()), 4),
            ambiguous_frac=round(float((inc0 & inc1).mean()), 4),
        ))
    return pd.DataFrame(rows)

def psd2_indicator(p_cal, y_cal, p_test, y_test, alpha_sig=0.05):
    """Indicateur de taux de fraude sur transactions approuvées, calculé
       ENTIÈREMENT sous la règle conforme (pas de mélange avec tau*)."""
    s_cal = nonconformity(p_cal, y_cal); n = len(s_cal)
    lvl = min(np.ceil((n+1)*(1-alpha_sig))/n, 1.0)
    q   = float(np.quantile(s_cal, lvl, method="higher"))
    flagged  = (1 - p_test) <= q                 # ensemble contenant "fraude"
    approved = ~flagged
    n_app = int(approved.sum())
    return dict(alpha_sig=alpha_sig, q_hat=round(q, 6),
                n_approved=n_app,
                frauds_among_approved=int(y_test[approved].sum()) if n_app else 0,
                lambda_pct=round(100*float(y_test[approved].mean()), 6) if n_app else np.nan,
                alert_volume_pct=round(100*float(flagged.mean()), 4))

print("✓ étage 4 défini (C1 + C2)")

# ============================================================================
#  CELLULE 7bis — Conforme conditionnel à la classe (Mondrian),
#  arbitrage couverture/alertes, sauvegarde des scores calibrés
# ============================================================================

def save_scores(dataset, seed, p_cal, y_cal, p_test, y_test):
    """Sauve les probabilités calibrées : toute analyse conforme ultérieure
       se refait alors en secondes, sans réentraîner."""
    d = OUT / "scores"; d.mkdir(exist_ok=True)
    np.savez_compressed(d / f"{dataset}_seed{seed}.npz",
                        p_cal=p_cal, y_cal=y_cal, p_test=p_test, y_test=y_test)


def mondrian_conformal(p_cal, y_cal, p_test, y_test, alphas=None, cfg=CFG):
    """Conforme conditionnel à la classe : un quantile par classe.

    C'est la variante qu'exige la Corollaire 5.14 telle qu'elle est écrite dans
    le manuscrit. La couverture marginale du Théorème 5.13 donne
    P(y ∉ Γ) ≤ α ; elle ne donne PAS P(1 ∉ Γ | y = 1) ≤ α. Le facteur π₁ de la
    Corollaire 5.14 suppose la seconde, qui demande un quantile calculé sur les
    seules fraudes de calibration.
    """
    alphas = cfg["ALPHA_SIG"] if alphas is None else alphas
    s_cal = nonconformity(p_cal, y_cal)
    s_f, s_l = s_cal[y_cal == 1], s_cal[y_cal == 0]
    n_f, n_l = len(s_f), len(s_l)
    rows = []
    for a in alphas:
        lv_f = min(np.ceil((n_f + 1) * (1 - a)) / n_f, 1.0) if n_f else 1.0
        lv_l = min(np.ceil((n_l + 1) * (1 - a)) / n_l, 1.0) if n_l else 1.0
        q_f = float(np.quantile(s_f, lv_f, method="higher")) if n_f else 1.0
        q_l = float(np.quantile(s_l, lv_l, method="higher")) if n_l else 1.0
        inc1 = (1 - p_test) <= q_f          # "fraude" dans l'ensemble
        inc0 = p_test <= q_l               # "légitime" dans l'ensemble
        cov = np.where(y_test == 1, inc1, inc0)
        m = y_test == 1
        rows.append(dict(
            alpha_sig=a, n_cal_fraud=int(n_f), q_fraud=round(q_f, 6),
            q_legit=round(q_l, 6),
            cov_fraud_class=round(float(cov[m].mean()), 4) if m.any() else np.nan,
            cov_legit_class=round(float(cov[~m].mean()), 4),
            cov_marginal=round(float(cov.mean()), 4),
            fraud_ok=bool(m.any() and cov[m].mean() >= 1 - a),
            alert_volume_pct=round(100 * float(inc1.mean()), 4),
            ambiguous_frac=round(float((inc0 & inc1).mean()), 4),
        ))
    return pd.DataFrame(rows)


def coverage_price(p_cal, y_cal, p_test, y_test, grid=None):
    """Le prix de la garantie : pour chaque niveau de couverture des fraudes
       visé, quel volume d'alertes faut-il accepter ?

       C'est le chiffre qu'une banque demande, et qu'aucune version du
       manuscrit ne donne."""
    grid = np.arange(0.50, 1.00, 0.05) if grid is None else grid
    s_f = nonconformity(p_cal, y_cal)[y_cal == 1]
    n_f = len(s_f)
    rows = []
    for target in grid:
        a = 1 - target
        lv = min(np.ceil((n_f + 1) * target) / n_f, 1.0) if n_f else 1.0
        q = float(np.quantile(s_f, lv, method="higher")) if n_f else 1.0
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


# ---------------------------------------------------------------------------
#  Agrégation — à exécuter après la cellule 11
# ---------------------------------------------------------------------------
def report_mondrian(ALL, out=None):
    out = OUT if out is None else out
    mon = pd.concat([pd.DataFrame(r["mondrian"]).assign(dataset=r["dataset"], seed=r["seed"])
                     for v in ALL.values() for r in v], ignore_index=True)
    trd = pd.concat([pd.DataFrame(r["tradeoff"]).assign(dataset=r["dataset"], seed=r["seed"])
                     for v in ALL.values() for r in v], ignore_index=True)
    mon.to_csv(out/"tables"/"conformal_mondrian.csv", index=False)
    trd.to_csv(out/"tables"/"coverage_price.csv", index=False)

    print("=== Conforme conditionnel à la classe (Mondrian) ===")
    g = mon.groupby(["dataset", "alpha_sig"])[
        ["cov_fraud_class", "cov_marginal", "alert_volume_pct"]].mean().round(4)
    print(g.to_string())
    ok = mon.groupby(["dataset", "alpha_sig"])["fraud_ok"].mean()
    print("\nfraud_ok (proportion de graines) :"); print(ok.to_string())

    print("\n=== Le prix de la garantie ===")
    t = trd.groupby(["dataset", "target_fraud_coverage"])[
        ["achieved_fraud_coverage", "alert_volume_pct", "lambda_approved_pct"]].mean().round(4)
    print(t.to_string())
    return mon, trd

print("✓ cellule 7bis chargée — Mondrian, arbitrage couverture/alertes, sauvegarde des scores")


# ============================================================================
#  CELLULE 8 — Étage 3 : SemiSync
# ============================================================================
def semisync(S, seed, cfg=CFG, max_iter=None):
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
                        alpha=list(np.round(a, 4)), ec_star=round(gate["ec_star"], 6)))
        state = (trio, gate, bst, Xa, raw)
        if prev is not None and abs(phi - prev) < cfg["SS_TOL"]:
            break
        prev, contam = phi, new
    df = pd.DataFrame(log)
    phis = df["phi"].values
    df.attrs["monotone"] = bool(np.all(np.diff(phis) >= -1e-9) or
                                np.all(np.diff(phis) <= 1e-9))
    df.attrs["oscillates"] = not df.attrs["monotone"]
    return df, state

print("✓ étage 3 défini")

# ============================================================================
#  CELLULE 9 — Un jeu de données, une graine
# ============================================================================
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             matthews_corrcoef)

def metrics_at(p, y, tau):
    pred = (p >= tau).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
    return dict(tau=round(float(tau), 4),
                precision=round(precision_score(y, pred, zero_division=0), 4),
                recall=round(recall_score(y, pred, zero_division=0), 4),
                f1=round(f1_score(y, pred, zero_division=0), 4),
                mcc=round(matthews_corrcoef(y, pred) if len(set(pred))>1 else 0.0, 4),
                auprc=round(average_precision_score(y, p), 4),
                roc_auc=round(roc_auc_score(y, p), 4),
                TP=tp, FP=fp, FN=fn, TN=tn,
                raw_cost=int(CFG["C_FP"]*fp + CFG["C_FN"]*fn))

def run_one(name, S, seed, cfg=CFG):
    t0 = time.time(); res = dict(dataset=name, seed=seed)
    res["splits"] = {k: dict(n=int(len(S[k][1])), frauds=int(S[k][1].sum()))
                     for k in S}

    ss_log, state = semisync(S, seed)
    trio, gate, bst, Xa, raw = state
    res["semisync"] = ss_log.to_dict("records")
    res["semisync_monotone"] = ss_log.attrs["monotone"]
    res["gate"] = {k: (list(map(float, v)) if isinstance(v, np.ndarray) else float(v))
                   for k, v in gate.items()}

    yv, yc, yt = S["val"][1], S["cal"][1], S["test"][1]
    models = {}

    # 1) ADAPTIVE-CP-FRAUD
    cname, p_val, others, ece_in, ece_out = calibrate(
        raw["val"], yv, {"cal": (raw["cal"], yc), "test": (raw["test"], yt)})
    tau, _ = best_tau(p_val, yv)                       # C5 : seuil sur val
    models["ADAPTIVE-CP-FRAUD"] = dict(
        **metrics_at(others["test"], yt, tau), calibrator=cname,
        ece_in_sample_val=round(ece_in, 5),
        ece_out_of_sample_test=round(ece_out["test"], 5))   # C3
    p_cal_adp, p_te_adp = others["cal"], others["test"]

    # 2) Baseline SPW — sans variables d'anomalie
    b2 = train_booster(S["train"][0], S["train"][1], S["val"][0], yv,
                       seed, use_cost=False)
    r2 = {k: b2.predict(xgb.DMatrix(S[k][0])) for k in S}
    _, pv2, o2, _, e2 = calibrate(r2["val"], yv,
                                  {"cal": (r2["cal"], yc), "test": (r2["test"], yt)})
    t2, _ = best_tau(pv2, yv)
    models["Baseline SPW"] = dict(**metrics_at(o2["test"], yt, t2),
                                  ece_out_of_sample_test=round(e2["test"], 5))

    # 3) HybridMeta-XGB — anomalies, poids uniformes, scale_pos_weight
    au = np.array([1/3, 1/3, 1/3])
    Xu = {k: augment(S[k][0], trio[k], au) for k in S}
    b3 = train_booster(Xu["train"], S["train"][1], Xu["val"], yv,
                       seed, use_cost=False)
    r3 = {k: b3.predict(xgb.DMatrix(Xu[k])) for k in S}
    _, pv3, o3, _, e3 = calibrate(r3["val"], yv,
                                  {"cal": (r3["cal"], yc), "test": (r3["test"], yt)})
    t3, _ = best_tau(pv3, yv)
    models["HybridMeta-XGB"] = dict(**metrics_at(o3["test"], yt, t3),
                                    ece_out_of_sample_test=round(e3["test"], 5))
    res["models"] = models

    # étage 4 — sur le bloc cal
    res["conformal"] = conformal(p_cal_adp, yc, p_te_adp, yt).to_dict("records")
    res["psd2"] = [psd2_indicator(p_cal_adp, yc, p_te_adp, yt, a)
                   for a in cfg["ALPHA_SIG"]]
    res["mondrian"] = mondrian_conformal(p_cal_adp, yc, p_te_adp, yt).to_dict("records")
    res["tradeoff"] = coverage_price(p_cal_adp, yc, p_te_adp, yt).to_dict("records")
    save_scores(name, seed, p_cal_adp, yc, p_te_adp, yt)

    # ablations (bloc test, seuil issu de val)
    abl = []
    for lbl, alpha_used, use_cost in [("full", gate["alpha"], True),
                                      ("no_stage1_uniform", au, True),
                                      ("no_stage2_spw", gate["alpha"], False)]:
        Xz = {k: augment(S[k][0], trio[k], alpha_used) for k in S}
        bz = train_booster(Xz["train"], S["train"][1], Xz["val"], yv, seed, use_cost)
        rz = {k: bz.predict(xgb.DMatrix(Xz[k])) for k in S}
        _, pvz, oz, _, _ = calibrate(rz["val"], yv, {"test": (rz["test"], yt)})
        tz, _ = best_tau(pvz, yv)
        abl.append(dict(config=lbl, **metrics_at(oz["test"], yt, tz)))
    abl.append(dict(config="baseline_no_anomaly", **models["Baseline SPW"]))
    res["ablation"] = abl

    res["runtime_min"] = round((time.time() - t0)/60, 2)
    print(f"  [{name} seed={seed}] {res['runtime_min']} min | "
          f"alpha*={gate['alpha']} | AUPRC={models['ADAPTIVE-CP-FRAUD']['auprc']}")
    return res

print("✓ orchestration définie")

# ============================================================================
#  CELLULE 10 — Exécution
# ============================================================================
ALL = {}

def run_dataset(name, X, y, extra_test=None, cfg=CFG):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    S = four_way(X, y)
    if extra_test is not None:                    # Dataset K, mode "provider"
        Xte, yte = extra_test
        sc = StandardScaler().fit(X[:int(len(y)*cfg['FRAC_TRAIN'])])
        S["test"] = (sc.transform(Xte), yte)
    split_df = report_split(name, S)
    split_df.to_csv(OUT/"tables"/f"table02_splits_{name}.csv", index=False)
    runs = [run_one(name, S, s) for s in cfg["SEEDS"]]
    ALL[name] = runs
    return runs

# --- jeux de données exécutés ----------------------------------------------
XT, yT = load_T(CFG["PATH_T"])
run_dataset("T", XT, yT)

XK, yK, extra = load_K(CFG["PATH_K_TRAIN"], CFG["PATH_K_TEST"])
run_dataset("K", XK, yK, extra_test=extra)

# IEEE-CIS (Dataset V) — ÉCARTÉ.
# Environ 28 h par graine : 10 graines dépasseraient onze jours de calcul.
# Le budget va au multi-graines sur T et K. Conséquence pour le manuscrit :
# voir la cellule markdown ci-dessus.
# XV, yV = load_V(CFG["PATH_V"]); run_dataset("V", XV, yV)

with open(OUT/"all_results.json", "w", encoding="utf-8") as f:
    json.dump(ALL, f, indent=2, default=str)
print(f"\n✓ écrit {OUT/'all_results.json'}")

# ============================================================================
#  CELLULE 11 — Agrégation et MANUSCRIPT_VALUES.md
# ============================================================================
def aggregate(runs):
    rows = []
    for r in runs:
        for m, v in r["models"].items():
            rows.append(dict(dataset=r["dataset"], seed=r["seed"], model=m, **v))
    return pd.DataFrame(rows)

def summarise(df, cols=("auprc","roc_auc","f1","mcc","precision","recall","raw_cost")):
    g = df.groupby(["dataset","model"])[list(cols)]
    out = g.agg(["mean","std"]).round(4)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()

# tabulate est optionnel : repli sur to_string s'il est absent
def _md(df, index=False):
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return df.to_string(index=index)

lines = ["# Valeurs à reporter dans le manuscrit",
         "", f"Graines : {CFG['SEEDS']}  |  découpage "
         f"{CFG['FRAC_TRAIN']:.0%}/{CFG['FRAC_VAL']:.0%}/"
         f"{CFG['FRAC_CAL']:.0%}/reste", ""]

if ALL:
    full = pd.concat([aggregate(v) for v in ALL.values()], ignore_index=True)
    full.to_csv(OUT/"tables"/"table10_main_all_seeds.csv", index=False)
    summ = summarise(full); summ.to_csv(OUT/"tables"/"table10_main_summary.csv", index=False)
    lines += ["## Tableau 10 — performance au seuil optimal (moyenne ± écart-type)",
              "", _md(summ), ""]

    conf = pd.concat([pd.DataFrame(r["conformal"]).assign(dataset=r["dataset"], seed=r["seed"])
                      for v in ALL.values() for r in v], ignore_index=True)
    conf.to_csv(OUT/"tables"/"table13_conformal.csv", index=False)
    lines += ["## Tableau 13 — couverture conforme",
              "",
              "`cov_fraud_class` est la couverture conditionnelle à la classe fraude :",
              "c'est elle qui porte l'argument sur les faux négatifs, et le manuscrit",
              "ne la distingue pas de la couverture marginale.", "",
              _md(conf), ""]

    ks = conf.groupby("dataset")[["ks_marginal","ks_fraud"]].mean().round(4)
    vac = conf.groupby("dataset")["infl_is_vacuous"].any()
    lines += ["## Statistique KS — Corollaire 5.16", "",
              "Les runs de l'ancien code donnaient ε̂ ≈ 0,83 avec une comparaison entre",
              "populations non comparables ; le manuscrit reporte 0,0933 / 0,099 / 0,0899.",
              "Valeurs recalculées :", "", _md(ks, index=True), "",
              "Borne d'inflation vide (α + 2ε̂ ≥ 0,5) : " + vac.to_dict().__str__(), ""]

    psd = pd.concat([pd.DataFrame(r["psd2"]).assign(dataset=r["dataset"], seed=r["seed"])
                     for v in ALL.values() for r in v], ignore_index=True)
    psd.to_csv(OUT/"tables"/"psd2_indicator.csv", index=False)
    lines += ["## Indicateur PSD2 — §8.3", "",
              "Calculé entièrement sous la règle conforme, avec le volume d'alertes",
              "correspondant. Le manuscrit combinait α_sig avec le FPR du seuil τ*=0,200,",
              "qui relève d'une autre règle de décision.", "", _md(psd), ""]

    abl = pd.concat([pd.DataFrame(r["ablation"]).assign(dataset=r["dataset"], seed=r["seed"])
                     for v in ALL.values() for r in v], ignore_index=True)
    abl.to_csv(OUT/"tables"/"table15_ablation.csv", index=False)
    lines += ["## Tableau 15 — ablation", "", _md(abl), ""]

    ece_rows = [dict(dataset=r["dataset"], seed=r["seed"], model=m,
                     ece_out_of_sample=v.get("ece_out_of_sample_test"))
                for v_ in ALL.values() for r in v_ for m, v in r["models"].items()]
    ece_df = pd.DataFrame(ece_rows); ece_df.to_csv(OUT/"tables"/"table06_ece.csv", index=False)
    lines += ["## Tableau 6 — ECE hors échantillon", "",
              "Remplace « ECE = 0,0000 on all models and datasets », qui mesurait",
              "l'ajustement isotonic sur son propre bloc.", "",
              _md(ece_df.groupby(["dataset","model"])["ece_out_of_sample"]
                    .agg(["mean","std"]).round(5), index=True), ""]

    ss = pd.concat([pd.DataFrame(r["semisync"]).assign(dataset=r["dataset"], seed=r["seed"])
                    for v in ALL.values() for r in v], ignore_index=True)
    ss.to_csv(OUT/"tables"/"table12_semisync.csv", index=False)
    mono = {d: [r["semisync_monotone"] for r in v] for d, v in ALL.items()}
    lines += ["## Tableaux 12 et 19 — SemiSync", "",
              f"Trajectoire monotone par jeu et par graine : {mono}", "",
              "Là où la trajectoire oscille, l'hypothèse de monotonie du Théorème 5.10",
              "ne tient pas et l'étage 3 doit être présenté comme une heuristique de",
              "calibration de la contamination.", ""]

(OUT/"MANUSCRIPT_VALUES.md").write_text("\n".join(lines), encoding="utf-8")
print(f"✓ écrit {OUT/'MANUSCRIPT_VALUES.md'}")
print("\n".join(lines[:40]))

# ============================================================================
#  CELLULE 11bis — Stabilité inter-graines
# ============================================================================
def seed_stability(ALL):
    rows, alpha_rows = [], []
    for ds, runs in ALL.items():
        # --- alpha* par graine
        for r in runs:
            a = r["gate"]["alpha"]
            alpha_rows.append(dict(dataset=ds, seed=r["seed"],
                                   alpha=str([round(float(x), 3) for x in a]),
                                   vertex=("IF" if a[0] > 0.99 else
                                           "LOF" if a[1] > 0.99 else
                                           "OC" if a[2] > 0.99 else "interior"),
                                   ec_star=round(float(r["gate"]["ec_star"]), 5),
                                   gain_pct=round(float(r["gate"]["gain_pct"]), 3)))
        # --- dispersion des métriques et stabilité du signe de delta
        for m in ("ADAPTIVE-CP-FRAUD", "Baseline SPW", "HybridMeta-XGB"):
            v = np.array([r["models"][m]["auprc"] for r in runs], float)
            rows.append(dict(dataset=ds, model=m, n_seeds=len(v),
                             auprc_mean=round(v.mean(), 4),
                             auprc_sd=round(v.std(ddof=1), 4) if len(v) > 1 else np.nan,
                             auprc_min=round(v.min(), 4), auprc_max=round(v.max(), 4),
                             cv_pct=round(100*v.std(ddof=1)/v.mean(), 2) if len(v) > 1 else np.nan))
    A = pd.DataFrame(alpha_rows); M = pd.DataFrame(rows)

    # --- signe de ΔAUPRC vs chaque comparateur, graine par graine
    sign_rows = []
    for ds, runs in ALL.items():
        for comp in ("Baseline SPW", "HybridMeta-XGB"):
            d = np.array([r["models"]["ADAPTIVE-CP-FRAUD"]["auprc"]
                          - r["models"][comp]["auprc"] for r in runs], float)
            sign_rows.append(dict(dataset=ds, comparator=comp,
                                  delta_mean=round(d.mean(), 4),
                                  delta_sd=round(d.std(ddof=1), 4) if len(d) > 1 else np.nan,
                                  n_positive=int((d > 0).sum()), n_seeds=len(d),
                                  sign_stable=bool((d > 0).all() or (d < 0).all())))
    Sg = pd.DataFrame(sign_rows)
    return A, M, Sg

if ALL:
    A, M, Sg = seed_stability(ALL)
    A.to_csv(OUT/"tables"/"seed_stability_alpha.csv", index=False)
    M.to_csv(OUT/"tables"/"seed_stability_metrics.csv", index=False)
    Sg.to_csv(OUT/"tables"/"seed_stability_signs.csv", index=False)

    print("\n--- alpha* par graine ---")
    print(A.groupby(["dataset", "vertex"]).size().to_string())
    print("\n--- dispersion AUPRC ---");   print(M.to_string(index=False))
    print("\n--- stabilité du signe ---"); print(Sg.to_string(index=False))

    unstable_a = A.groupby("dataset")["vertex"].nunique()
    for ds, k in unstable_a.items():
        if k > 1:
            print(f"\n  !! {ds} : alpha* change de sommet selon la graine "
                  f"({k} sommets distincts). La dépendance à la topologie doit "
                  f"être requalifiée pour ce jeu.")
    for _, r in Sg.iterrows():
        if not r["sign_stable"]:
            print(f"\n  !! {r['dataset']} vs {r['comparator']} : le signe de "
                  f"ΔAUPRC change selon la graine ({r['n_positive']}/"
                  f"{r['n_seeds']} positifs). À écrire comme indiscernable du "
                  f"bruit d'initialisation, pas comme un écart non significatif.")


# ============================================================================
#  CELLULE 12 — Tests statistiques
# ============================================================================
from scipy import stats

def paired_bootstrap(y, pa, pb, reps=None, seed=42):
    reps = CFG["BOOTSTRAP_REPS"] if reps is None else reps
    g = rng_for(seed, "bootstrap"); n = len(y)
    obs = average_precision_score(y, pa) - average_precision_score(y, pb)
    cnt, diffs = 0, []
    for _ in range(reps):
        i = g.integers(0, n, n)
        if y[i].sum() == 0: continue
        d = average_precision_score(y[i], pa[i]) - average_precision_score(y[i], pb[i])
        diffs.append(d); cnt += (d * obs <= 0)
    return dict(delta_auprc=round(float(obs), 4),
                p_bootstrap=round(float(cnt/max(len(diffs), 1)), 4),
                ci_low=round(float(np.percentile(diffs, 2.5)), 4),
                ci_high=round(float(np.percentile(diffs, 97.5)), 4))

def holm(pvals, labels):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(run, 1.0)
    return {labels[i]: round(float(adj[i]), 4) for i in range(m)}

print("✓ tests définis — appliquer holm() aux p-valeurs des trois jeux de données")

# ============================================================================
#  CELLULE — PaySim (mobile money), troisième jeu de données
#
#  À exécuter APRÈS les cellules 1 à 9 (les fonctions doivent être en mémoire).
#  Écrit les .npz, puis les cellules 12 et 13 intègrent PaySim automatiquement.
# ============================================================================

CFG["FILE_PS"]  = "paysim.csv"
CFG["PATH_PS"]  = str(Path(CFG["DATA_DIR"]) / CFG["FILE_PS"])

# 6,3 M de lignes : le scoring OC-SVM et LOF domine le temps de calcul.
# None = tout le fichier. Un entier = les N dernières lignes chronologiques.
PS_MAX_ROWS = None          # mettre 2_000_000 pour un premier essai
PS_SEEDS    = [42, 43, 44]  # 3 graines suffisent pour la forme de la courbe

assert os.path.exists(CFG["PATH_PS"]), f"introuvable : {CFG['PATH_PS']}"
print(f"PaySim : {os.path.getsize(CFG['PATH_PS'])/1e6:.0f} Mo")


def load_PS(path, max_rows=None):
    """PaySim — transactions de mobile money simulées.

    Choix de variables, et pourquoi :

    * `isFlaggedFraud` est ÉCARTÉ. C'est le drapeau de la règle interne du
      simulateur (elle se déclenche sur les TRANSFER au-dessus d'un seuil).
      L'inclure revient à donner au modèle la sortie d'un oracle partiel :
      c'est une fuite, et un relecteur la verrait immédiatement.
    * `step` sert au tri chronologique mais n'est PAS une variable. Sur un
      découpage chronologique dont le test est postérieur, un index temporel
      en entrée fait apprendre la position dans le temps plutôt que la fraude
      — même raison que pour `unix_time` sur Sparkov.
    * `nameOrig` / `nameDest` sont des identifiants à cardinalité quasi unique.
      On n'en garde que l'information utile : le destinataire est-il un
      marchand (préfixe M) ou un compte client (préfixe C).
    * `type` est encodé en indicatrices : la fraude n'apparaît que dans deux
      des cinq types, c'est le signal le plus fort du jeu.
    * Les deux écarts de solde sont dérivés parce qu'ils encodent
      l'incohérence comptable qui caractérise les transferts frauduleux.
    """
    df = pd.read_csv(path)
    need = {"step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFraud"}
    missing = need - set(df.columns)
    assert not missing, f"colonnes absentes : {missing}\nprésentes : {list(df.columns)}"

    df = df.sort_values("step", kind="mergesort").reset_index(drop=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)
        print(f"  sous-échantillon chronologique : {len(df):,} dernières lignes")

    y = df["isFraud"].astype(int).values

    X = pd.DataFrame(index=df.index)
    X["amount"]         = df["amount"].astype(np.float32)
    X["log_amount"]     = np.log1p(df["amount"].values).astype(np.float32)
    X["oldbalanceOrg"]  = df["oldbalanceOrg"].astype(np.float32)
    X["newbalanceOrig"] = df["newbalanceOrig"].astype(np.float32)
    X["oldbalanceDest"] = df["oldbalanceDest"].astype(np.float32)
    X["newbalanceDest"] = df["newbalanceDest"].astype(np.float32)
    # incohérences comptables
    X["errBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]
                           - df["oldbalanceOrg"]).astype(np.float32)
    X["errBalanceDest"] = (df["oldbalanceDest"] + df["amount"]
                           - df["newbalanceDest"]).astype(np.float32)
    # destinataire marchand ou client
    if "nameDest" in df.columns:
        X["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M") \
                                  .astype(np.float32)
    # type de transaction, en indicatrices
    for t in sorted(df["type"].astype(str).unique()):
        X[f"type_{t}"] = (df["type"].astype(str) == t).astype(np.float32)

    print(f"  {len(y):,} transactions | {int(y.sum()):,} fraudes "
          f"({100*y.mean():.4f} %) | {X.shape[1]} variables")
    print(f"  variables : {list(X.columns)}")
    print("  écartées : isFlaggedFraud (fuite), step (index temporel), "
          "nameOrig/nameDest (identifiants)")
    return X.values.astype(np.float32), y


# ---------------------------------------------------------------------------
XPS, yPS = load_PS(CFG["PATH_PS"], PS_MAX_ROWS)

_seeds_backup = CFG["SEEDS"]
CFG["SEEDS"] = PS_SEEDS
try:
    run_dataset("PS", XPS, yPS)          # remplit ALL["PS"] et écrit les .npz
finally:
    CFG["SEEDS"] = _seeds_backup

with open(OUT/"all_results.json", "w", encoding="utf-8") as f:
    json.dump(ALL, f, indent=2, default=str)
print("\njeux de données dans ALL :", {k: len(v) for k, v in ALL.items()})
print("npz :", sorted(os.listdir(OUT/"scores")))
print("\n-> relancer les cellules 12 et 13 : PaySim y sera intégré automatiquement.")


# ============================================================================
#  CELLULE UNIQUE — tous les résultats du papier simplifié
#
#  Prérequis dans la session : CFG, OUT, ALL, np, pd  (+ les .npz écrits par
#  save_scores). Ne réentraîne rien : quelques secondes.
#
#  Produit :  PAPER_RESULTS.md
#             tables/paper_*.csv
#             figures/fig_coverage_price.png   (figure centrale)
# ============================================================================
import json, glob
from pathlib import Path
from scipy.stats import ks_2samp

OUT = Path(CFG["OUT_DIR"]); (OUT/"tables").mkdir(parents=True, exist_ok=True)
(OUT/"figures").mkdir(exist_ok=True)

# ---------------------------------------------------------------- helpers
def _s(p, y):                      # score de non-conformité, Déf. 5.11
    return np.where(y == 1, 1 - p, p)

def _q(scores, level):
    n = len(scores)
    if n == 0: return 1.0
    lv = min(np.ceil((n + 1) * level) / n, 1.0)
    return float(np.quantile(scores, lv, method="higher"))

def load_scores():
    """{dataset: {seed: (p_cal, y_cal, p_test, y_test)}}"""
    out = {}
    for f in sorted(glob.glob(str(OUT/"scores"/"*.npz"))):
        stem = Path(f).stem                       # ex. T_seed42
        ds, sd = stem.split("_seed")
        z = np.load(f)
        out.setdefault(ds, {})[int(sd)] = (z["p_cal"], z["y_cal"],
                                           z["p_test"], z["y_test"])
    return out

S = load_scores()
if not S:
    raise SystemExit(
        "Aucun .npz dans " + str(OUT/"scores") + "\n"
        "-> ajoute save_scores(...) dans run_one() et relance la cellule 10.")
print("scores chargés :", {k: sorted(v) for k, v in S.items()})

ALPHAS = CFG["ALPHA_SIG"]
TARGETS = np.round(np.arange(0.50, 1.00, 0.05), 2)

# ================================================================ BLOC A
#  Couverture marginale vs conditionnelle à la classe
# ========================================================================
rows = []
for ds, per_seed in S.items():
    for sd, (pc, yc, pt, yt) in per_seed.items():
        sc, st = _s(pc, yc), _s(pt, yt)
        m = yt == 1
        ks_marg = float(ks_2samp(sc, st).statistic)
        ks_frd  = float(ks_2samp(sc[yc == 1], st[m]).statistic) if m.any() else np.nan
        for a in ALPHAS:
            # --- marginal : un seul quantile sur tout le bloc cal
            qm = _q(sc, 1 - a)
            cov_m = np.where(m, (1 - pt) <= qm, pt <= qm)
            # --- Mondrian : un quantile par classe
            qf, ql = _q(sc[yc == 1], 1 - a), _q(sc[yc == 0], 1 - a)
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
A = pd.DataFrame(rows)
A.to_csv(OUT/"tables"/"paper_A_coverage.csv", index=False)

# ================================================================ BLOC C
#  Le prix de la garantie
# ========================================================================
rows = []
for ds, per_seed in S.items():
    for sd, (pc, yc, pt, yt) in per_seed.items():
        sf = _s(pc, yc)[yc == 1]
        m = yt == 1
        for t in TARGETS:
            q = _q(sf, t)
            fl = (1 - pt) <= q
            ap = ~fl
            rows.append(dict(
                dataset=ds, seed=sd, target_fraud_coverage=float(t),
                achieved_fraud_coverage=round(float(fl[m].mean()), 4),
                alert_volume_pct=round(100 * float(fl.mean()), 4),
                n_alerts=int(fl.sum()), frauds_caught=int(fl[m].sum()),
                lambda_approved_pct=round(100 * float(yt[ap].mean()), 6) if ap.any() else np.nan,
                base_rate_pct=round(100 * float(yt.mean()), 6)))
C = pd.DataFrame(rows)
C.to_csv(OUT/"tables"/"paper_C_price.csv", index=False)

# ================================================================ SECONDAIRE
#  Modèle, ablation, gate, SemiSync — depuis ALL
# ========================================================================
mod = pd.DataFrame([dict(dataset=r["dataset"], seed=r["seed"], model=k,
                         auprc=v["auprc"], raw_cost=v["raw_cost"],
                         precision=v["precision"], recall=v["recall"],
                         mcc=v["mcc"], ece=v.get("ece_out_of_sample_test"))
                    for runs in ALL.values() for r in runs
                    for k, v in r["models"].items()])
mod.to_csv(OUT/"tables"/"paper_models.csv", index=False)

abl = pd.DataFrame([dict(dataset=r["dataset"], seed=r["seed"], **a)
                    for runs in ALL.values() for r in runs for a in r["ablation"]])
abl.to_csv(OUT/"tables"/"paper_ablation.csv", index=False)

gate = pd.DataFrame([dict(dataset=r["dataset"], seed=r["seed"],
                          alpha=str([round(float(x), 3) for x in r["gate"]["alpha"]]),
                          vertex=("IF" if r["gate"]["alpha"][0] > .99 else
                                  "LOF" if r["gate"]["alpha"][1] > .99 else
                                  "OC" if r["gate"]["alpha"][2] > .99 else "interior"),
                          gain_pct=round(float(r["gate"]["gain_pct"]), 2))
                     for runs in ALL.values() for r in runs])
gate.to_csv(OUT/"tables"/"paper_gate.csv", index=False)

ss = pd.DataFrame([dict(dataset=r["dataset"], seed=r["seed"],
                        n_iter=len(r["semisync"]),
                        phi_unique=len({round(x["phi"], 6) for x in r["semisync"]}))
                   for runs in ALL.values() for r in runs])

# ================================================================ FIGURE
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ds, sub in C.groupby("dataset"):
        g = sub.groupby("target_fraud_coverage")["alert_volume_pct"]
        mu, sd = g.mean(), g.std()
        ax.plot(mu.index, mu.values, marker="o", label=f"Dataset {ds}")
        ax.fill_between(mu.index, mu - sd, mu + sd, alpha=0.18)
    ax.set_xlabel("Target fraud-class coverage $1-\\alpha$")
    ax.set_ylabel("Alert volume (% of transactions)")
    ax.set_title("The price of a class-conditional conformal guarantee")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"figures"/"fig_coverage_price.png", dpi=200)
    plt.close(fig); fig_ok = True
except Exception as e:
    fig_ok = False; print("figure non générée:", e)

# ================================================================ RAPPORT
def md(df, index=False):
    try:    return df.to_markdown(index=index)
    except Exception: return df.to_string(index=index)

L = ["# Résultats du papier simplifié", "",
     f"Graines : {sorted(next(iter(S.values())))}  |  "
     f"découpage {CFG['FRAC_TRAIN']:.0%}/{CFG['FRAC_VAL']:.0%}/{CFG['FRAC_CAL']:.0%}/reste", ""]

L += ["## A — Couverture marginale contre couverture conditionnelle", "",
      md(A.groupby(["dataset", "alpha_sig"])[
          ["marg_cov_marginal", "marg_cov_fraud",
           "mond_cov_fraud", "mond_alert_pct"]].mean().round(4), index=True), "",
      "Proportion de graines où la couverture des fraudes atteint la cible :", "",
      md(A.groupby(["dataset", "alpha_sig"])[
          ["marg_fraud_ok", "mond_fraud_ok"]].mean().round(2), index=True), ""]

L += ["## C — Le prix de la garantie", "",
      md(C.groupby(["dataset", "target_fraud_coverage"])[
          ["achieved_fraud_coverage", "alert_volume_pct",
           "lambda_approved_pct", "base_rate_pct"]].mean().round(4), index=True), ""]

L += ["## Statistique KS", "",
      md(A.groupby("dataset")[["ks_marginal", "ks_fraud"]].mean().round(4), index=True), ""]

L += ["## Modèle et comparateurs", "",
      md(mod.groupby(["dataset", "model"])[["auprc", "raw_cost", "ece"]]
            .agg(["mean", "std"]).round(4), index=True), ""]

L += ["## Ablation — coût brut", "",
      md(abl.groupby(["dataset", "config"])["raw_cost"]
            .agg(["mean", "std"]).round(1), index=True), ""]
for ds, sub in abl.groupby("dataset"):
    piv = sub.pivot_table(index="seed", columns="config", values="raw_cost")
    if "full" in piv:
        w = {c: int((piv[c] < piv["full"]).sum()) for c in piv.columns if c != "full"}
        L += [f"Graines où la configuration bat le pipeline complet — {ds} : {w} sur {len(piv)}", ""]

L += ["## Résultats négatifs", "",
      "Sommet de $\\alpha^*$ par jeu et par graine :", "",
      md(gate.groupby(["dataset", "vertex"]).size().rename("n").reset_index()), "",
      "SemiSync — itérations et nombre de valeurs distinctes de $\\phi$ :", "",
      md(ss.groupby("dataset")[["n_iter", "phi_unique"]].agg(["mean", "max"]).round(2), index=True), ""]

(OUT/"PAPER_RESULTS.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\n✓", OUT/"PAPER_RESULTS.md", "| figure:", fig_ok)


# ============================================================================
#  CELLULE 13 (version rapide) — Le prix contre la taille de calibration
#
#  Même calcul que la précédente, mais les scores de test sont triés UNE fois
#  et interrogés par recherche dichotomique au lieu d'être reparcourus à chaque
#  itération. Passe de plusieurs heures à quelques secondes sur PaySim.
# ============================================================================
import glob, time
import numpy as np, pandas as pd
from pathlib import Path

OUT = Path(CFG["OUT_DIR"])
(OUT/"tables").mkdir(parents=True, exist_ok=True); (OUT/"figures").mkdir(exist_ok=True)

N_GRID   = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 250, 500, 1000]
N_RESAMP = 40
TARGETS  = [0.80, 0.90, 0.95]

def _q(sc, lv):
    n = len(sc)
    if n == 0: return 1.0
    return float(np.quantile(sc, min(np.ceil((n + 1) * lv) / n, 1.0), method="higher"))

rows = []
for f in sorted(glob.glob(str(OUT/"scores"/"*.npz"))):
    t0 = time.time()
    ds, sd = Path(f).stem.split("_seed"); sd = int(sd)
    z = np.load(f)
    pc, yc, pt, yt = z["p_cal"], z["y_cal"], z["p_test"], z["y_test"]

    # --- préparation : un tri, puis tout se lit par dichotomie -------------
    s_test_all   = np.sort(1.0 - pt)                    # tout le trafic
    s_test_fraud = np.sort((1.0 - pt)[yt == 1])         # les fraudes
    n_te, n_fr = len(s_test_all), len(s_test_fraud)
    s_f_all = np.sort(1.0 - pc[yc == 1])                # fraudes de calibration
    n_all = len(s_f_all)
    g = np.random.default_rng(10_000 + sd)

    def counts(q):
        """(volume d'alertes, fraudes attrapées) en O(log n)."""
        return (np.searchsorted(s_test_all,   q, side="right") / n_te,
                np.searchsorted(s_test_fraud, q, side="right") / max(n_fr, 1))

    sizes = [n for n in N_GRID if n < n_all] + [n_all]
    for n_f in sizes:
        for rep in range(1 if n_f == n_all else N_RESAMP):
            s_f = s_f_all if n_f == n_all else \
                  np.sort(s_f_all[g.choice(n_all, n_f, replace=False)])
            for t in TARGETS:
                q = _q(s_f, t)
                av, ach = counts(q)
                rows.append(dict(dataset=ds, seed=sd, n_cal_fraud=n_f,
                                 resample=rep, target=t,
                                 achieved=round(float(ach), 4),
                                 alert_pct=round(100 * float(av), 4),
                                 degenerate=bool(av > 0.5)))
    print(f"  {ds} seed={sd} : {n_all} fraudes de calibration, "
          f"{n_te:,} points de test — {time.time()-t0:.1f} s")

D = pd.DataFrame(rows)
D.to_csv(OUT/"tables"/"paper_D_ncal_sweep.csv", index=False)

summ = (D.groupby(["dataset", "target", "n_cal_fraud"])
          .agg(alert_median=("alert_pct", "median"),
               alert_p10=("alert_pct", lambda x: np.percentile(x, 10)),
               alert_p90=("alert_pct", lambda x: np.percentile(x, 90)),
               achieved_mean=("achieved", "mean"),
               p_degenerate=("degenerate", "mean"))
          .round(4).reset_index())
print("\n", summ.to_string(index=False))

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAMES = {"T": "ULB", "K": "Sparkov", "PS": "PaySim"}
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(13, 4), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), TARGETS):
        for ds in sorted(summ.dataset.unique()):
            s = summ[(summ.target == t) & (summ.dataset == ds)].sort_values("n_cal_fraud")
            ax.plot(s.n_cal_fraud, s.alert_median, marker="o", label=NAMES.get(ds, ds))
            ax.fill_between(s.n_cal_fraud, s.alert_p10, s.alert_p90, alpha=.15)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Frauds in calibration split")
        ax.set_title(f"Target fraud coverage {t:.2f}"); ax.grid(alpha=.3, which="both")
    np.atleast_1d(axes)[0].set_ylabel("Alert volume (%)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/"figures"/"fig_ncal_price.png", dpi=200)
    plt.close(fig); print("\n✓ figures/fig_ncal_price.png")
except Exception as e:
    print("figure non générée:", e)


# ============================================================================
#  CELLULE — Figures 1 et 4 du papier
#
#  Recalculées depuis les .npz : quelques secondes, aucun réentraînement.
#  S'adaptent aux jeux présents (T, K, PS…).
#
#  Produit : figures/fig_coverage_gap.png    — la dissociation des couvertures
#            figures/fig_mechanism.png       — pourquoi le prix explose
#            tables/paper_E_gap.csv
# ============================================================================
import glob
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(CFG["OUT_DIR"])
(OUT/"figures").mkdir(exist_ok=True); (OUT/"tables").mkdir(exist_ok=True)

NAMES   = {"T": "ULB", "K": "Sparkov", "PS": "PaySim"}
ALPHAS  = CFG["ALPHA_SIG"]
TARGETS = [0.75, 0.90, 0.95]

def _s(p, y):  return np.where(y == 1, 1 - p, p)
def _q(sc, lv):
    n = len(sc)
    if n == 0: return 1.0
    return float(np.quantile(sc, min(np.ceil((n + 1) * lv) / n, 1.0), method="higher"))

# ---------------------------------------------------------------- lecture
S = {}
for f in sorted(glob.glob(str(OUT/"scores"/"*.npz"))):
    ds, sd = Path(f).stem.split("_seed")
    z = np.load(f)
    S.setdefault(ds, {})[int(sd)] = (z["p_cal"], z["y_cal"], z["p_test"], z["y_test"])
order = [d for d in ("T", "K", "PS") if d in S] + [d for d in S if d not in ("T","K","PS")]
print("jeux :", {d: len(S[d]) for d in order})

# ================================================================ FIGURE 1
#  Dissociation : couverture marginale contre couverture de la classe fraude
# ========================================================================
rows = []
for ds in order:
    for sd, (pc, yc, pt, yt) in S[ds].items():
        sc, m = _s(pc, yc), (yt == 1)
        for a in ALPHAS:
            qm = _q(sc, 1 - a)
            cov_m = np.where(m, (1 - pt) <= qm, pt <= qm)
            qf, ql = _q(sc[yc == 1], 1 - a), _q(sc[yc == 0], 1 - a)
            cov_c = np.where(m, (1 - pt) <= qf, pt <= ql)
            rows.append(dict(dataset=ds, seed=sd, alpha=a,
                             marg_all=cov_m.mean(), marg_fraud=cov_m[m].mean(),
                             cond_fraud=cov_c[m].mean()))
G = pd.DataFrame(rows)
G.to_csv(OUT/"tables"/"paper_E_gap.csv", index=False)
g = G.groupby(["dataset", "alpha"]).agg(["mean", "std"])

fig, axes = plt.subplots(1, len(order), figsize=(4.6*len(order), 4.0), sharey=True)
for ax, ds in zip(np.atleast_1d(axes), order):
    x = np.arange(len(ALPHAS)); w = 0.27
    sub = g.loc[ds]
    for k, (col, lab, c) in enumerate([
            ("marg_all",   "marginal predictor: overall coverage",      "#4C78A8"),
            ("marg_fraud", "marginal predictor: fraud-class coverage",  "#E45756"),
            ("cond_fraud", "class-conditional: fraud-class coverage",   "#59A14F")]):
        ax.bar(x + (k-1)*w, sub[(col, "mean")].values, w, label=lab, color=c,
               yerr=sub[(col, "std")].values, capsize=2, error_kw=dict(lw=.8))
    for i, a in enumerate(ALPHAS):
        ax.hlines(1-a, i-1.6*w, i+1.6*w, ls="--", lw=1.2, color="black", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([f"$\\alpha={a}$" for a in ALPHAS])
    ax.set_ylim(0, 1.05); ax.set_title(NAMES.get(ds, ds)); ax.grid(axis="y", alpha=.3)
np.atleast_1d(axes)[0].set_ylabel("Empirical coverage")
np.atleast_1d(axes)[0].legend(fontsize=8, loc="center left", framealpha=.95)
fig.suptitle("Marginal coverage is met; the frauds it covers are not "
             "(dashed line: nominal target $1-\\alpha$)", y=1.0, fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"figures"/"fig_coverage_gap.png", dpi=200,
                                bbox_inches="tight")
plt.close(fig)

# ================================================================ FIGURE 4
#  Mécanisme : lire le prix directement sur les deux fonctions de répartition
# ========================================================================
fig, axes = plt.subplots(1, len(order), figsize=(4.8*len(order), 4.2), sharey=True)
mech = []
for ax, ds in zip(np.atleast_1d(axes), order):
    sd = sorted(S[ds])[0]
    pc, yc, pt, yt = S[ds][sd]
    s_f  = np.sort(_s(pc, yc)[yc == 1])          # fraudes, bloc calibration
    s_lg = np.sort((1 - pt)[yt == 0])            # légitimes, bloc test
    ax.plot(s_f,  np.arange(1, len(s_f)+1)/len(s_f),   lw=2, color="#E45756",
            label=f"calibration frauds ($n={len(s_f)}$)")
    ax.plot(s_lg, np.arange(1, len(s_lg)+1)/len(s_lg), lw=2, color="#4C78A8",
            label="test legitimate traffic")
    for t, col in zip(TARGETS, ["#999999", "#666666", "#111111"]):
        q  = _q(s_f, t)
        av = float(((1 - pt) <= q).mean())
        ax.vlines(q, 0, t, ls=":", lw=1.3, color=col)
        ax.hlines(t, 0, q, ls=":", lw=1.3, color=col)
        ax.annotate(f"{t:.0%} → {100*av:.2g}% alerts", xy=(q, t),
                    xytext=(6, -11), textcoords="offset points",
                    fontsize=8, color=col)
        mech.append(dict(dataset=ds, seed=sd, target=t, q=q, alert_pct=100*av))
    ax.set_xlabel("Nonconformity score  $s = 1-\\hat{p}$")
    ax.set_title(NAMES.get(ds, ds)); ax.grid(alpha=.3); ax.set_xlim(0, 1.02)
np.atleast_1d(axes)[0].set_ylabel("Empirical CDF")
np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Reading the price: the quantile that covers the last frauds sits "
             "inside the legitimate mass", y=1.0, fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"figures"/"fig_mechanism.png", dpi=200,
                                bbox_inches="tight")
plt.close(fig)

print("\n--- dissociation (moyenne sur les graines) ---")
print(g[[("marg_all","mean"),("marg_fraud","mean"),("cond_fraud","mean")]].round(4).to_string())
print("\n--- lecture du mécanisme ---")
print(pd.DataFrame(mech).round(4).to_string(index=False))
print("\n✓ figures/fig_coverage_gap.png  et  figures/fig_mechanism.png")
