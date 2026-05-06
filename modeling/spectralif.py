"""
spectralif.py
=============
SpectralIF: Two-Detector Fusion with Validation-Optimized Alpha

Architecture
------------
SpectralIF trains TWO separate Isolation Forests:
  1. IF_base   — trained on kinematic features:
                 (altitude, velocity, RSS, Doppler, alt/vel ratio)
  2. IF_spec   — trained on LSNF spectral features:
                 (entropy, flatness, dominant_freq, BER, rolloff, centroid)

The final anomaly score fuses both detectors:

    A(x) = alpha * hat_s_base(x) + (1 - alpha) * hat_s_spec(x)

where hat_s denotes min-max normalisation using training score
statistics (no leakage), and alpha in [0,1] is optimised on a
held-out validation fold to maximise F1.

Decision Rule
-------------
    anomaly if A(x) >= threshold*

where threshold* is jointly optimised with alpha on the validation set.

Why Two Detectors?
------------------
A single IF over all features dilutes the spectral signal — kinematic
features (velocity, altitude) dominate the isolation splits because they
have higher variance. Separating the feature spaces forces the second
detector to learn the spectral structure independently, then fusion
combines complementary decision boundaries.

This is the key methodological contribution of SpectralIF over naive
feature concatenation.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from typing import Tuple, Optional, Dict

from processing.signal_processing import LSNFTransformer, SPECTRAL_FEATURE_NAMES


class SpectralIF:
    """
    SpectralIF detector.

    Parameters
    ----------
    k              : LSNF neighbourhood size (default 8, optimised by ablation)
    contamination  : expected anomaly fraction (default 0.20)
    n_estimators   : trees per Isolation Forest (default 200)
    val_size       : fraction of training data used for alpha/threshold optimisation
    alpha_grid     : candidate alpha values for grid search
    thresh_grid    : candidate threshold values for grid search
    random_state   : reproducibility seed
    """

    def __init__(
        self,
        k: int = 8,
        contamination: float = 0.20,
        n_estimators: int = 200,
        val_size: float = 0.20,
        alpha_grid: Optional[np.ndarray] = None,
        thresh_grid: Optional[np.ndarray] = None,
        random_state: int = 42,
    ):
        self.k             = k
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.val_size      = val_size
        self.alpha_grid    = alpha_grid if alpha_grid is not None else np.arange(0.0, 1.01, 0.10)
        self.thresh_grid   = thresh_grid if thresh_grid is not None else np.arange(0.05, 0.95, 0.05)
        self.random_state  = random_state

        # Fitted attributes
        self.alpha_     : float = 0.90
        self.threshold_ : float = 0.20
        self.lsnf_      : Optional[LSNFTransformer] = None
        self.sc_base_   : Optional[StandardScaler]  = None
        self.sc_spec_   : Optional[StandardScaler]  = None
        self.if_base_   : Optional[IsolationForest] = None
        self.if_spec_   : Optional[IsolationForest] = None
        self._tr_sb     : Optional[np.ndarray] = None
        self._tr_ss     : Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y_val_hint: Optional[np.ndarray] = None):
        """
        Fit SpectralIF on training data.

        Parameters
        ----------
        X           : ndarray (n, n_base_features) — kinematic features only
        y_val_hint  : optional binary labels for alpha optimisation.
                      If provided, a stratified inner split is used.
                      If None, contamination-based threshold is applied.
        """
        # Inner train / validation split for alpha optimisation
        if y_val_hint is not None:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_val_hint,
                test_size=self.val_size,
                stratify=y_val_hint,
                random_state=self.random_state,
            )
        else:
            X_tr, X_val = train_test_split(
                X, test_size=self.val_size, random_state=self.random_state
            )
            y_val = None

        # LSNF — fit on X_tr only
        self.lsnf_ = LSNFTransformer(
            k=min(self.k, len(X_tr) - 1),
            coord_cols=[0, 1], signal_col=1,
        )
        X_tr_s  = self.lsnf_.fit_transform(X_tr)
        X_val_s = self.lsnf_.transform(X_val)

        # Scalers
        self.sc_base_ = StandardScaler().fit(X_tr)
        self.sc_spec_ = StandardScaler().fit(X_tr_s)
        X_tr_b  = self.sc_base_.transform(X_tr);  X_val_b = self.sc_base_.transform(X_val)
        X_tr_sp = self.sc_spec_.transform(X_tr_s); X_val_sp = self.sc_spec_.transform(X_val_s)

        # Two Isolation Forests
        self.if_base_ = IsolationForest(
            n_estimators=self.n_estimators, contamination=self.contamination,
            random_state=self.random_state, n_jobs=-1,
        ).fit(X_tr_b)

        self.if_spec_ = IsolationForest(
            n_estimators=self.n_estimators, contamination=self.contamination,
            random_state=self.random_state, n_jobs=-1,
        ).fit(X_tr_sp)

        # Store training scores for min-max normalisation reference
        self._tr_sb = self.if_base_.score_samples(X_tr_b)
        self._tr_ss = self.if_spec_.score_samples(X_tr_sp)

        # Optimise alpha and threshold on validation fold
        val_sb = self.if_base_.score_samples(X_val_b)
        val_ss = self.if_spec_.score_samples(X_val_sp)
        val_bn = self._minmax(-self._tr_sb, -val_sb)
        val_sn = self._minmax(-self._tr_ss, -val_ss)

        if y_val is not None:
            best_f1, best_a, best_t = 0, 0.90, 0.20
            for alpha in self.alpha_grid:
                h_val = alpha * val_bn + (1 - alpha) * val_sn
                for thresh in self.thresh_grid:
                    preds = (h_val >= thresh).astype(int)
                    _, _, f1, _ = precision_recall_fscore_support(
                        y_val, preds, average="binary", zero_division=0
                    )
                    if f1 > best_f1:
                        best_f1, best_a, best_t = f1, alpha, thresh
            self.alpha_     = best_a
            self.threshold_ = best_t
        else:
            # Fallback: use contamination percentile
            h_val = 0.90 * val_bn + 0.10 * val_sn
            self.alpha_     = 0.90
            self.threshold_ = float(np.percentile(h_val, (1 - self.contamination) * 100))

        return self

    # ------------------------------------------------------------------
    def _minmax(self, ref: np.ndarray, x: np.ndarray) -> np.ndarray:
        lo, hi = ref.min(), ref.max()
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def _hybrid_score(self, X: np.ndarray) -> np.ndarray:
        X_s  = self.lsnf_.transform(X)
        X_b  = self.sc_base_.transform(X)
        X_sp = self.sc_spec_.transform(X_s)
        sb   = self.if_base_.score_samples(X_b)
        ss   = self.if_spec_.score_samples(X_sp)
        bn   = self._minmax(-self._tr_sb, -sb)
        sn   = self._minmax(-self._tr_ss, -ss)
        return self.alpha_ * bn + (1 - self.alpha_) * sn

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous."""
        return self._hybrid_score(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns -1 (anomaly) or +1 (normal), matching sklearn convention."""
        scores = self._hybrid_score(X)
        return np.where(scores >= self.threshold_, -1, 1)

    def predict_proba_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Hybrid score as anomaly probability proxy (for AUC-ROC)."""
        return self._hybrid_score(X)

    def get_params(self) -> Dict:
        return {
            "alpha":     self.alpha_,
            "threshold": self.threshold_,
            "k":         self.k,
            "contamination": self.contamination,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.attack_injection import build_benchmark_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

    print("Building dataset...")
    df = build_benchmark_dataset(n_legit=1500, n_pm=180, n_ghost=180, n_vd=180, n_replay=150)
    df['altitude_velocity_ratio'] = df['baro_altitude'] / df['velocity'].clip(lower=1.0)

    FEAT = [c for c in ['baro_altitude','velocity','rss','doppler_shift','altitude_velocity_ratio']
            if c in df.columns]
    X = df[FEAT].fillna(df[FEAT].mean()).values.astype(float)
    y = (df['label'].values > 0).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    print("Fitting SpectralIF...")
    clf = SpectralIF(k=8, contamination=0.20)
    clf.fit(X_tr, y_val_hint=y_tr)
    print(f"Optimised params: {clf.get_params()}")

    preds  = clf.predict(X_te)
    scores = clf.score_samples(X_te)
    y_pred = (preds == -1).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_te, scores)
    print(f"Test: P={p:.3f}  R={r:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    print("SpectralIF smoke test passed ✓")