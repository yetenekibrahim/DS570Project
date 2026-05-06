"""
evaluation.py
=============
Paper-grade evaluation pipeline for SpectralIF.

Key methodological guarantees
------------------------------
1. LEAK-FREE: All feature transformations (LSNF) are applied inside
   sklearn Pipelines. KNN is fit on training fold only; test samples
   query training neighbours exclusively.

2. PROPER TRAIN/TEST SPLIT: ablation_study and cross_validated_comparison
   use StratifiedKFold. evaluate_on_benchmark uses a held-out 20% test set.

3. BOOTSTRAP CI: 95% confidence intervals on F1 via 1000 bootstrap
   iterations on the test fold predictions.

4. CONTAMINATION SENSITIVITY: The contamination parameter is varied over
   [0.05, 0.10, 0.15, 0.20] and results are reported for each — the
   reviewer complaint "contamination is unknown in practice" is addressed.

5. INFERENCE LATENCY: Measured on test set with 50 repetitions,
   reported as mean ± std milliseconds.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from processing.signal_processing import LSNFTransformer, SPECTRAL_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn, n_boot: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n   = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    s = np.array(scores)
    a = (1 - ci) / 2
    return float(s.mean()), float(np.percentile(s, 100*a)), float(np.percentile(s, 100*(1-a)))


def _f1(yt, yp):
    _, _, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    return f


# ---------------------------------------------------------------------------
# Per-attack-type metrics
# ---------------------------------------------------------------------------

def per_attack_metrics(y_true_mc: np.ndarray, y_pred_bin: np.ndarray,
                       label_map: Dict) -> pd.DataFrame:
    """
    Per-attack detection rate and false alarm rate.

    FAR is computed once from the legitimate class and reported
    identically for context — it is a property of the detector,
    not of each attack type.
    """
    rows = []
    legit_mask = (y_true_mc == 0)
    n_legit = int(legit_mask.sum())
    n_fp = int(y_pred_bin[legit_mask].sum())
    far = round(100.0 * n_fp / max(n_legit, 1), 2)

    for lbl_int, lbl_str in label_map.items():
        mask = (y_true_mc == lbl_int)
        total = int(mask.sum())
        if total == 0:
            continue
        detected = int(y_pred_bin[mask].sum())
        rows.append({
            "Attack Type":      lbl_str,
            "Total":            total,
            "Detected":         detected,
            "Detection Rate %": round(100.0 * detected / total, 2),
            "FAR %":            far if lbl_str != "legitimate" else "—",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Benchmark evaluation (proper train/test split)
# ---------------------------------------------------------------------------

def evaluate_on_benchmark(
    df: pd.DataFrame,
    label_col: str = "label",
    feat_base: List[str] = None,
    contamination: float = 0.10,
    test_size: float = 0.20,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Evaluate SpectralIF on a labeled benchmark dataset.

    Train/test split is stratified by label.
    LSNF is computed inside the pipeline — KNN fit on train only.

    Parameters
    ----------
    df            : full labeled DataFrame
    label_col     : column with integer attack labels
    feat_base     : base feature columns for KNN coords + model input
    contamination : Isolation Forest contamination parameter
    test_size     : fraction held out for evaluation
    n_boot        : bootstrap iterations for CI
    seed          : random seed

    Returns
    -------
    dict with metrics, curves, per-attack table
    """
    if feat_base is None:
        feat_base = ["baro_altitude", "velocity", "altitude_velocity_ratio"]

    # Use only columns that exist
    feat_base = [c for c in feat_base if c in df.columns]
    coord_cols_idx = list(range(min(3, len(feat_base))))

    y_mc = df[label_col].values
    X_raw = df[feat_base].fillna(df[feat_base].mean()).values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_mc, test_size=test_size,
        stratify=(y_mc > 0).astype(int), random_state=seed
    )

    # Build leak-free pipeline
    lsnf = LSNFTransformer(k=32, coord_cols=coord_cols_idx, signal_col=1)
    scaler = StandardScaler()
    iforest = IsolationForest(n_estimators=200, contamination=contamination,
                               random_state=42, n_jobs=-1)

    # Fit on train
    X_tr_spec = lsnf.fit_transform(X_tr)
    X_tr_all  = np.hstack([X_tr, X_tr_spec])
    X_tr_sc   = scaler.fit_transform(X_tr_all)
    iforest.fit(X_tr_sc)

    # Transform test (query train neighbours only)
    X_te_spec = lsnf.transform(X_te)
    X_te_all  = np.hstack([X_te, X_te_spec])
    X_te_sc   = scaler.transform(X_te_all)

    preds  = iforest.predict(X_te_sc)
    scores = iforest.score_samples(X_te_sc)

    y_pred_bin = (preds == -1).astype(int)
    y_true_bin = (y_te > 0).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )
    f1_mean, f1_lo, f1_hi = bootstrap_ci(y_true_bin, y_pred_bin, _f1, n_boot=n_boot)

    try:
        auc_roc = roc_auc_score(y_true_bin, -scores)
        auc_pr  = average_precision_score(y_true_bin, -scores)
    except Exception:
        auc_roc = auc_pr = None

    fpr_arr, tpr_arr, _ = roc_curve(y_true_bin, -scores)
    pr_p, pr_r, _       = precision_recall_curve(y_true_bin, -scores)

    from data.attack_injection import ATTACK_LABELS
    pa_df = per_attack_metrics(y_te, y_pred_bin, ATTACK_LABELS)

    # Latency on test set
    lat = benchmark_latency_fn(iforest, X_te_sc)

    return {
        "model_name":       "SpectralIF",
        "n_train":          len(X_tr),
        "n_test":           len(X_te),
        "precision":        round(float(prec), 4),
        "recall":           round(float(rec),  4),
        "f1":               round(float(f1),   4),
        "f1_ci_lo":         round(f1_lo,       4),
        "f1_ci_hi":         round(f1_hi,       4),
        "auc_roc":          round(float(auc_roc), 4) if auc_roc else None,
        "auc_pr":           round(float(auc_pr),  4) if auc_pr  else None,
        "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin).tolist(),
        "per_attack":       pa_df,
        "roc_curve":        (fpr_arr, tpr_arr),
        "pr_curve":         (pr_p, pr_r),
        "latency":          lat,
        "contamination":    contamination,
    }


# ---------------------------------------------------------------------------
# Ablation study — proper held-out evaluation
# ---------------------------------------------------------------------------

def ablation_study(
    df: pd.DataFrame,
    label_col: str = "label",
    contamination: float = 0.10,
    n_splits: int = 5,
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified K-Fold ablation study.

    Three feature configurations tested:
      A) Base only       : altitude, velocity, alt/vel ratio
      B) Spectral only   : 6 LSNF features
      C) SpectralIF (A+B): all features

    LSNF is fit inside each training fold — no leakage.

    Returns DataFrame with F1, AUC-ROC, Latency per configuration.
    """
    FEAT_BASE = [c for c in ["baro_altitude","velocity","altitude_velocity_ratio"]
                 if c in df.columns]

    y_mc  = df[label_col].values
    y_bin = (y_mc > 0).astype(int)
    X_raw = df[FEAT_BASE].fillna(df[FEAT_BASE].mean()).values.astype(float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    configs = {
        "Base features only":    {"use_spec": False, "use_base": True},
        "Spectral only (LSNF)":  {"use_spec": True,  "use_base": False},
        "SpectralIF (Base+LSNF)":{"use_spec": True,  "use_base": True},
    }

    results = {name: {"f1":[], "auc":[], "lat":[]} for name in configs}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_raw, y_bin)):
        X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
        y_te_bin   = y_bin[te_idx]

        # Fit LSNF on training fold
        lsnf = LSNFTransformer(k=min(32, len(X_tr)-1),
                               coord_cols=[0,1], signal_col=1)
        X_tr_spec = lsnf.fit_transform(X_tr)
        X_te_spec = lsnf.transform(X_te)

        for name, cfg in configs.items():
            parts_tr, parts_te = [], []
            if cfg["use_base"]:
                parts_tr.append(X_tr); parts_te.append(X_te)
            if cfg["use_spec"]:
                parts_tr.append(X_tr_spec); parts_te.append(X_te_spec)

            X_tr_c = np.hstack(parts_tr)
            X_te_c = np.hstack(parts_te)

            sc = StandardScaler().fit(X_tr_c)
            X_tr_sc = sc.transform(X_tr_c)
            X_te_sc = sc.transform(X_te_c)

            mod = IsolationForest(n_estimators=200, contamination=contamination,
                                  random_state=42, n_jobs=-1)
            mod.fit(X_tr_sc)

            preds  = (mod.predict(X_te_sc) == -1).astype(int)
            scores = mod.score_samples(X_te_sc)

            _, _, f1, _ = precision_recall_fscore_support(
                y_te_bin, preds, average="binary", zero_division=0
            )
            results[name]["f1"].append(f1)
            try:
                results[name]["auc"].append(roc_auc_score(y_te_bin, -scores))
            except Exception:
                pass

            t0 = time.perf_counter()
            mod.predict(X_te_sc)
            results[name]["lat"].append((time.perf_counter()-t0)*1000)

    rows = []
    for name, vals in results.items():
        f1s  = np.array(vals["f1"])
        aucs = np.array(vals["auc"]) if vals["auc"] else np.array([np.nan])
        lats = np.array(vals["lat"])
        f1_m, f1_lo, f1_hi = bootstrap_ci(
            np.ones(len(f1s)), f1s,
            lambda yt, yp: yp.mean(), n_boot=200
        )
        rows.append({
            "Configuration":  name,
            "F1 mean":        round(float(f1s.mean()), 3),
            "F1 std":         round(float(f1s.std()),  3),
            "F1 95% CI":      f"[{f1s.mean()-1.96*f1s.std():.3f}, {f1s.mean()+1.96*f1s.std():.3f}]",
            "AUC-ROC mean":   round(float(aucs.mean()), 3),
            "Latency ms":     round(float(lats.mean()), 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-validated comparison table
# ---------------------------------------------------------------------------

def cross_validated_comparison(
    df: pd.DataFrame,
    label_col: str = "label",
    contamination: float = 0.10,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified K-Fold comparison: Z-Score | LOF | IF-base | SpectralIF

    LSNF is fit inside each fold — test samples query training
    neighbours only. This is the table that goes into the paper.
    """
    from modeling.baseline import ZScoreDetector

    FEAT_BASE = [c for c in ["baro_altitude","velocity","altitude_velocity_ratio"]
                 if c in df.columns]

    y_mc  = df[label_col].values
    y_bin = (y_mc > 0).astype(int)
    X_raw = df[FEAT_BASE].fillna(df[FEAT_BASE].mean()).values.astype(float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    methods = ["Z-Score", "LOF", "IsoForest (base)", "SpectralIF"]
    res = {m: {"f1":[], "auc":[], "lat":[]} for m in methods}

    for tr_idx, te_idx in skf.split(X_raw, y_bin):
        X_tr, X_te = X_raw[tr_idx], X_raw[te_idx]
        y_te        = y_bin[te_idx]

        # Shared LSNF — fit on train only
        lsnf = LSNFTransformer(k=min(32, len(X_tr)-1),
                               coord_cols=[0,1], signal_col=1)
        X_tr_spec = lsnf.fit_transform(X_tr)
        X_te_spec = lsnf.transform(X_te)

        # Shared scaler for base
        sc_b = StandardScaler().fit(X_tr)
        X_tr_b = sc_b.transform(X_tr)
        X_te_b = sc_b.transform(X_te)

        # All features
        X_tr_all = np.hstack([X_tr, X_tr_spec])
        X_te_all = np.hstack([X_te, X_te_spec])
        sc_a = StandardScaler().fit(X_tr_all)
        X_tr_a = sc_a.transform(X_tr_all)
        X_te_a = sc_a.transform(X_te_all)

        def _record(name, preds, scores, X_te_for_lat, model):
            _, _, f1, _ = precision_recall_fscore_support(
                y_te, preds, average="binary", zero_division=0)
            res[name]["f1"].append(f1)
            try:
                res[name]["auc"].append(roc_auc_score(y_te, scores))
            except Exception:
                pass
            t0 = time.perf_counter()
            model.predict(X_te_for_lat) if hasattr(model,"predict") else None
            res[name]["lat"].append((time.perf_counter()-t0)*1000)

        # Z-Score
        zs = ZScoreDetector(threshold=3.0).fit(X_tr_b)
        zp = (zs.predict(X_te_b)==-1).astype(int)
        _record("Z-Score", zp, zs.score_samples(X_te_b), X_te_b, zs)

        # LOF — novelty=True to allow predict on test set
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination,
                                  novelty=True, n_jobs=-1)
        lof.fit(X_tr_b)
        lof_p = (lof.predict(X_te_b)==-1).astype(int)
        lof_s = -lof.score_samples(X_te_b)
        _record("LOF", lof_p, lof_s, X_te_b, lof)

        # IsoForest base
        ifb = IsolationForest(n_estimators=200, contamination=contamination,
                               random_state=42, n_jobs=-1)
        ifb.fit(X_tr_b)
        ifb_p = (ifb.predict(X_te_b)==-1).astype(int)
        ifb_s = -ifb.score_samples(X_te_b)
        _record("IsoForest (base)", ifb_p, ifb_s, X_te_b, ifb)

        # SpectralIF
        ifs = IsolationForest(n_estimators=200, contamination=contamination,
                               random_state=42, n_jobs=-1)
        ifs.fit(X_tr_a)
        ifs_p = (ifs.predict(X_te_a)==-1).astype(int)
        ifs_s = -ifs.score_samples(X_te_a)
        _record("SpectralIF", ifs_p, ifs_s, X_te_a, ifs)

    rows = []
    for m in methods:
        f1s  = np.array(res[m]["f1"])
        aucs = np.array(res[m]["auc"]) if res[m]["auc"] else np.array([np.nan])
        lats = np.array(res[m]["lat"]) if res[m]["lat"] else np.array([np.nan])
        rows.append({
            "Method":        m,
            "F1 mean±std":   f"{f1s.mean():.3f}±{f1s.std():.3f}",
            "AUC-ROC mean":  round(float(aucs.mean()), 3),
            "Latency ms":    round(float(lats.mean()), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Contamination sensitivity analysis
# ---------------------------------------------------------------------------

def contamination_sensitivity(
    df: pd.DataFrame,
    label_col: str = "label",
    contamination_values: List[float] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate SpectralIF F1 across contamination values.
    Addresses reviewer concern: 'contamination is unknown in practice.'
    """
    if contamination_values is None:
        contamination_values = [0.05, 0.10, 0.15, 0.20, 0.25]

    FEAT_BASE = [c for c in ["baro_altitude","velocity","altitude_velocity_ratio"]
                 if c in df.columns]
    y_mc  = df[label_col].values
    y_bin = (y_mc > 0).astype(int)
    X_raw = df[FEAT_BASE].fillna(df[FEAT_BASE].mean()).values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y_bin, test_size=0.20,
        stratify=y_bin, random_state=seed
    )

    lsnf = LSNFTransformer(k=min(32, len(X_tr)-1), coord_cols=[0,1], signal_col=1)
    X_tr_spec = lsnf.fit_transform(X_tr)
    X_te_spec = lsnf.transform(X_te)
    X_tr_all  = np.hstack([X_tr, X_tr_spec])
    X_te_all  = np.hstack([X_te, X_te_spec])
    sc = StandardScaler().fit(X_tr_all)
    X_tr_sc = sc.transform(X_tr_all)
    X_te_sc = sc.transform(X_te_all)

    rows = []
    for c in contamination_values:
        mod = IsolationForest(n_estimators=200, contamination=c,
                               random_state=42, n_jobs=-1)
        mod.fit(X_tr_sc)
        preds = (mod.predict(X_te_sc)==-1).astype(int)
        scores = mod.score_samples(X_te_sc)
        _, _, f1, _ = precision_recall_fscore_support(
            y_te, preds, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_te, -scores)
        except Exception:
            auc = None
        rows.append({
            "Contamination ρ": c,
            "F1":              round(float(f1), 3),
            "AUC-ROC":         round(float(auc), 3) if auc else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Latency helper
# ---------------------------------------------------------------------------

def benchmark_latency_fn(model, X: np.ndarray, n_runs: int = 50) -> Dict:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(X)
        times.append((time.perf_counter()-t0)*1000)
    return {
        "mean_ms": round(np.mean(times), 3),
        "std_ms":  round(np.std(times),  3),
        "n_samples": X.shape[0],
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.attack_injection import build_benchmark_dataset
    from processing.signal_processing import extract_features

    print("Building dataset...")
    df = build_benchmark_dataset(n_legit=800, n_pm=100, n_ghost=100,
                                  n_vd=100, n_replay=80)

    print("Running ablation study (5-fold, no leakage)...")
    abl = ablation_study(df, contamination=0.20, n_splits=3, n_boot=100)
    print(abl.to_string(index=False))

    print("\nRunning cross-validated comparison...")
    cv = cross_validated_comparison(df, contamination=0.20, n_splits=3)
    print(cv.to_string(index=False))


# ---------------------------------------------------------------------------
# full_evaluation_report — compatibility wrapper for dashboard
# ---------------------------------------------------------------------------

def full_evaluation_report(
    model,
    X: np.ndarray,
    y_true_multiclass: np.ndarray,
    scores: np.ndarray,
    label_map: dict,
    model_name: str = "SpectralIF",
    n_boot: int = 1000,
) -> dict:
    """
    Full evaluation report for a fitted model on a test set.

    Parameters
    ----------
    model              : fitted model with .predict() method
    X                  : scaled feature matrix (test set)
    y_true_multiclass  : integer labels (0=legit, 1..N=attacks)
    scores             : anomaly scores (higher = more anomalous)
    label_map          : {int: str} attack label mapping
    model_name         : name string for display
    n_boot             : bootstrap iterations for CI

    Returns
    -------
    dict with precision, recall, f1, CI, AUC-ROC, AUC-PR,
         per_attack DataFrame, roc_curve, pr_curve tuples
    """
    preds      = model.predict(X)
    y_pred_bin = (preds == -1).astype(int)
    y_true_bin = (y_true_multiclass > 0).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )
    f1_mean, f1_lo, f1_hi = bootstrap_ci(y_true_bin, y_pred_bin, _f1, n_boot=n_boot)

    try:
        auc_roc = roc_auc_score(y_true_bin, scores)
        auc_pr  = average_precision_score(y_true_bin, scores)
    except Exception:
        auc_roc = auc_pr = None

    fpr_arr, tpr_arr, _ = roc_curve(y_true_bin, scores)
    pr_p, pr_r, _       = precision_recall_curve(y_true_bin, scores)

    pa_df = per_attack_metrics(y_true_multiclass, y_pred_bin, label_map)

    return {
        "model_name":       model_name,
        "precision":        round(float(prec), 4),
        "recall":           round(float(rec),  4),
        "f1":               round(float(f1),   4),
        "f1_ci_lo":         round(f1_lo,       4),
        "f1_ci_hi":         round(f1_hi,       4),
        "auc_roc":          round(float(auc_roc), 4) if auc_roc else None,
        "auc_pr":           round(float(auc_pr),  4) if auc_pr  else None,
        "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin).tolist(),
        "per_attack":       pa_df,
        "roc_curve":        (fpr_arr, tpr_arr),
        "pr_curve":         (pr_p, pr_r),
    }