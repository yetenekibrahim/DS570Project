"""
baseline.py
===========
Baseline anomaly detection methods for benchmarking against SpectralIF.

Methods
-------
1. Z-Score       — Univariate statistical baseline. An observation is
                   flagged as anomalous if any feature exceeds k standard
                   deviations from the mean (default k=3).

2. Local Outlier Factor (LOF) — Density-based method (Breunig et al.,
                   2000). Compares the local density of each point to
                   those of its neighbours. Computationally heavier
                   than IF but captures local cluster structures.

These two baselines, combined with a vanilla IF (no spectral features),
form the comparison set for the SpectralIF evaluation framework.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Z-Score Baseline
# ---------------------------------------------------------------------------

class ZScoreDetector:
    """
    Multivariate Z-Score anomaly detector.

    A sample x is anomalous if any feature satisfies |z_i| > threshold,
    where z_i = (x_i - μ_i) / σ_i.

    Parameters
    ----------
    threshold : float
        Number of standard deviations beyond which a point is anomalous.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-12
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the maximum absolute z-score across features.
        Higher = more anomalous.
        """
        z = np.abs((X - self.mean_) / self.std_)
        return z.max(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns -1 (anomaly) or +1 (normal), matching IF convention."""
        scores = self.score_samples(X)
        return np.where(scores > self.threshold, -1, 1)


# ---------------------------------------------------------------------------
# LOF Baseline
# ---------------------------------------------------------------------------

class LOFDetector:
    """
    Wrapper around sklearn LocalOutlierFactor for batch anomaly detection.

    LOF computes the local reachability density of each point relative
    to its k nearest neighbours; points in low-density regions (outliers)
    receive high LOF scores.

    Parameters
    ----------
    n_neighbors   : int   — neighbourhood size k (default 20).
    contamination : float — expected anomaly fraction.
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors   = n_neighbors
        self.contamination = contamination
        self._lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,
            n_jobs=-1,
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return predictions (-1 / +1) in one step."""
        return self._lof.fit_predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Negative LOF scores: more negative = more anomalous.
        Available only after fit_predict().
        """
        return self._lof.negative_outlier_factor_


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    contamination: float = 0.05,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Train all baseline methods plus SpectralIF and return a comparison table.

    Parameters
    ----------
    X            : feature matrix (already scaled)
    y_true       : ground-truth labels (0=normal, 1=attack); None = unsupervised
    contamination: expected anomaly fraction

    Returns
    -------
    pd.DataFrame with columns:
        Method | Precision | Recall | F1 | AUC-ROC | Anomaly_Rate
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import (
        precision_recall_fscore_support, roc_auc_score
    )

    results = []

    def _metrics(preds_binary, scores_for_auc, name):
        row = {"Method": name}
        row["Anomaly_Rate (%)"] = round(100 * preds_binary.mean(), 2)
        if y_true is not None:
            y_bin = (y_true > 0).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_bin, preds_binary, average="binary", zero_division=0
            )
            row["Precision"] = round(float(prec), 3)
            row["Recall"]    = round(float(rec),  3)
            row["F1"]        = round(float(f1),   3)
            try:
                row["AUC-ROC"] = round(
                    float(roc_auc_score(y_bin, scores_for_auc)), 3
                )
            except Exception:
                row["AUC-ROC"] = None
        return row

    # 1. Z-Score
    zs = ZScoreDetector(threshold=3.0).fit(X)
    zs_scores = zs.score_samples(X)
    zs_preds  = (zs.predict(X) == -1).astype(int)
    results.append(_metrics(zs_preds, zs_scores, "Z-Score (baseline)"))

    # 2. LOF
    lof = LOFDetector(n_neighbors=20, contamination=contamination)
    lof_raw   = lof.fit_predict(X)
    lof_preds = (lof_raw == -1).astype(int)
    lof_scores = -lof.score_samples(X)   # negate: higher = more anomalous
    results.append(_metrics(lof_preds, lof_scores, "LOF (baseline)"))

    # 3. Isolation Forest (no spectral features)
    n_base = min(3, X.shape[1])   # use only first n_base features
    if_base = IsolationForest(
        contamination=contamination, n_estimators=200,
        random_state=42, n_jobs=-1
    )
    if_base.fit(X[:, :n_base])
    if_base_scores = -if_base.score_samples(X[:, :n_base])
    if_base_preds  = (if_base.predict(X[:, :n_base]) == -1).astype(int)
    results.append(_metrics(if_base_preds, if_base_scores, "IsoForest (base features)"))

    # 4. SpectralIF (all features)
    if_full = IsolationForest(
        contamination=contamination, n_estimators=200,
        random_state=42, n_jobs=-1
    )
    if_full.fit(X)
    if_full_scores = -if_full.score_samples(X)
    if_full_preds  = (if_full.predict(X) == -1).astype(int)
    results.append(_metrics(if_full_preds, if_full_scores, "SpectralIF (proposed)"))

    df = pd.DataFrame(results)
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X  = np.vstack([rng.randn(450, 9), rng.randn(50, 9) * 4])
    y  = np.array([0] * 450 + [1] * 50)

    table = run_comparison(X, y_true=y, contamination=0.10)
    print(table.to_string(index=False))