"""
anomaly_detection.py
====================
SpectralIF: Isolation Forest augmented with frequency-domain features
and SHAP-based explainability for ADS-B anomaly detection.

Algorithm choice rationale
--------------------------
Isolation Forest (Liu et al., 2008) was selected over supervised
alternatives (LSTM, Transformer) and density-based methods (LOF) for
three reasons:

1. Label-free — no attack examples required at training time.
2. Scalable — O(n log n) training, O(log n) inference; handles the
   ~8,000 simultaneous flights returned by OpenSky without delay.
3. Anomaly-score transparency — raw path-length scores are monotone
   and directly interpretable with SHAP TreeExplainer.

The hybrid anomaly score A(x) combines the IF score with the
normalised spectral entropy:

    A(x) = α · s_IF(x)  +  (1 - α) · ŝ_entropy(x)

where α ∈ [0,1] is tuned on the benchmark dataset.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Feature columns used by the model
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "baro_altitude",
    "velocity",
    "altitude_velocity_ratio",
]

SPECTRAL_FEATURES = [
    "spectral_entropy",
    "spectral_flatness",
    "dominant_freq",
    "band_energy_ratio",
    "spectral_rolloff",
    "spectral_centroid",
]

ALL_FEATURES = BASE_FEATURES + SPECTRAL_FEATURES


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Select, clean, and standardise features for the Isolation Forest.

    StandardScaler is applied because path-length isolation is sensitive
    to feature scale: without normalisation, high-variance features
    (e.g. baro_altitude in metres) would dominate the split selection.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of str, optional
        Defaults to ALL_FEATURES (base + spectral).

    Returns
    -------
    X_scaled : ndarray, shape (n_samples, n_features)
    scaler   : fitted StandardScaler
    used_cols: list of feature column names actually used
    """
    if feature_cols is None:
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    used_cols = feature_cols

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, scaler, used_cols


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    """
    Fit an Isolation Forest.

    Hyperparameters
    ---------------
    n_estimators=200  — more trees than default (100) for stable scores
                        on large flight datasets.
    contamination     — expected anomaly fraction; set by the user via
                        the dashboard slider.
    random_state=42   — reproducibility.

    Returns
    -------
    Fitted IsolationForest instance.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def detect_anomalies(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Predict anomaly labels.  Returns 1 (normal) or -1 (anomaly).
    """
    return model.predict(X)


def get_anomaly_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Raw anomaly scores: more negative → more anomalous.
    Scores are the negative of the average path length normalised by
    the expected path length for a random dataset of the same size.
    """
    return model.score_samples(X)


# ---------------------------------------------------------------------------
# Hybrid anomaly score (SpectralIF contribution)
# ---------------------------------------------------------------------------

def hybrid_anomaly_score(
    if_scores: np.ndarray,
    spectral_entropy_values: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Combine IF path-length score with spectral entropy into a unified
    anomaly score (SpectralIF):

        A(x) = α · ŝ_IF(x)  +  (1 - α) · ŝ_entropy(x)

    Both components are min-max normalised to [0, 1] before fusion,
    so that the weighting α is interpretable as a mixing coefficient.

    Parameters
    ----------
    if_scores             : raw score_samples() output (more negative = worse)
    spectral_entropy_values : per-sample spectral entropy
    alpha                 : weight on IF score (default 0.7)

    Returns
    -------
    hybrid : ndarray in [0, 1] where higher = more anomalous
    """
    def _norm(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    # IF: invert so higher = more anomalous
    if_norm = _norm(-if_scores)
    # Entropy: higher entropy = more anomalous (already in same direction)
    ent_norm = _norm(spectral_entropy_values)

    return alpha * if_norm + (1.0 - alpha) * ent_norm


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    predictions: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute evaluation metrics.

    If y_true is provided (benchmark mode), full precision/recall/F1/AUC
    are computed.  Otherwise, descriptive statistics only are returned.

    Parameters
    ----------
    predictions : ndarray — IF predictions (+1 / -1)
    y_true      : ndarray — ground-truth binary labels (0=normal, 1=attack)
    scores      : ndarray — raw anomaly scores for AUC-ROC computation

    Returns
    -------
    dict with metric keys.
    """
    n_total    = len(predictions)
    n_anomaly  = int(np.sum(predictions == -1))
    n_normal   = int(np.sum(predictions == 1))
    anomaly_rate = round(100 * n_anomaly / n_total, 2)

    result = {
        "total":        n_total,
        "normal":       n_normal,
        "anomalies":    n_anomaly,
        "anomaly_rate": anomaly_rate,
    }

    if y_true is not None:
        # Convert IF convention (-1/+1) to binary (1/0 anomaly)
        y_pred_bin = (predictions == -1).astype(int)
        y_true_bin = (y_true > 0).astype(int)      # 0=legit, 1=any attack

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average="binary", zero_division=0
        )
        result["precision"] = round(float(prec), 4)
        result["recall"]    = round(float(rec),  4)
        result["f1_score"]  = round(float(f1),   4)

        if scores is not None:
            # AUC-ROC: higher score = more normal, so negate for convention
            try:
                auc = roc_auc_score(y_true_bin, -scores)
                result["auc_roc"] = round(float(auc), 4)
            except Exception:
                result["auc_roc"] = None

        result["confusion_matrix"] = confusion_matrix(
            y_true_bin, y_pred_bin
        ).tolist()

    return result


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: IsolationForest,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 500,
):
    """
    Compute SHAP values for the Isolation Forest using TreeExplainer.

    SHAP (SHapley Additive exPlanations) decomposes each prediction into
    additive feature contributions φ_i such that:

        f(x) = φ_0 + Σ_i φ_i(x)

    For the Isolation Forest, positive φ_i indicates that feature i
    pushes the score toward anomalous; negative φ_i pushes toward normal.

    Parameters
    ----------
    model        : fitted IsolationForest
    X            : ndarray, shape (n, p)
    feature_names: list of p feature names
    max_samples  : cap for computational efficiency

    Returns
    -------
    shap_values : ndarray, shape (min(n, max_samples), p)
    X_subset    : ndarray corresponding rows
    """
    try:
        import shap
        X_sub = X[:max_samples]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sub)
        return shap_values, X_sub
    except ImportError:
        print("[SHAP] shap package not installed — skipping explainability.")
        return None, None
    except Exception as e:
        print(f"[SHAP] Computation failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X_normal  = rng.randn(400, len(ALL_FEATURES))
    X_attack  = rng.randn(50,  len(ALL_FEATURES)) * 3 + 5
    X         = np.vstack([X_normal, X_attack])
    y         = np.array([0] * 400 + [1] * 50)

    model   = train_isolation_forest(X, contamination=0.10)
    preds   = detect_anomalies(model, X)
    scores  = get_anomaly_scores(model, X)
    metrics = evaluate_model(preds, y_true=y, scores=scores)

    print("Evaluation:", metrics)