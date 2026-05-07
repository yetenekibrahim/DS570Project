"""Unit tests for SpectralIF and baselines"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from modeling.spectralif import SpectralIF
from modeling.baseline import ZScoreDetector, LOFDetector
from modeling.evaluation import bootstrap_ci, per_attack_metrics

RNG = np.random.RandomState(7)

def _make_X(n=300, n_feat=5):
    return np.column_stack([
        RNG.uniform(0, 12000, n),   # altitude
        RNG.uniform(50, 400, n),    # velocity
        RNG.normal(-85, 8, n),      # rss
        RNG.normal(0, 3, n),        # doppler
        RNG.uniform(10, 200, n),    # alt/vel ratio
    ])


class TestSpectralIF:
    def test_fit_predict(self):
        X = _make_X(200)
        y = np.array([0]*160 + [1]*40)
        clf = SpectralIF(k=8, contamination=0.20)
        clf.fit(X, y_val_hint=y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_score_samples_shape(self):
        X = _make_X(200)
        y = np.array([0]*160 + [1]*40)
        clf = SpectralIF(k=8, contamination=0.20)
        clf.fit(X, y_val_hint=y)
        scores = clf.score_samples(X)
        assert scores.shape == (200,)

    def test_scores_in_01(self):
        X = _make_X(200)
        y = np.array([0]*160 + [1]*40)
        clf = SpectralIF(k=8, contamination=0.20)
        clf.fit(X, y_val_hint=y)
        scores = clf.score_samples(X)
        assert scores.min() >= -0.01
        assert scores.max() <= 1.01

    def test_alpha_optimised(self):
        X = _make_X(300)
        y = np.array([0]*240 + [1]*60)
        clf = SpectralIF(k=8, contamination=0.20)
        clf.fit(X, y_val_hint=y)
        assert 0.0 <= clf.alpha_ <= 1.0
        assert 0.0 <= clf.threshold_ <= 1.0

    def test_no_leakage_test_not_in_train(self):
        X = _make_X(300)
        y = np.array([0]*240 + [1]*60)
        X_tr, X_te = X[:240], X[240:]
        y_tr = y[:240]
        clf = SpectralIF(k=8, contamination=0.20)
        clf.fit(X_tr, y_val_hint=y_tr)
        # Should predict without error on unseen test set
        preds = clf.predict(X_te)
        assert len(preds) == 60


class TestZScoreDetector:
    def test_flags_clear_outliers(self):
        X = np.vstack([RNG.randn(100, 5), np.full((10, 5), 10)])
        zs = ZScoreDetector(threshold=3.0).fit(X)
        preds = zs.predict(X)
        assert (preds[-10:] == -1).all()

    def test_low_fpr_on_normal(self):
        X = RNG.randn(1000, 5)
        zs = ZScoreDetector(threshold=5.0).fit(X)
        rate = (zs.predict(X) == -1).mean()
        assert rate < 0.01


class TestBootstrapCI:
    def test_ci_bounds(self):
        y_true = np.array([0,0,1,1,1,0,1,0])
        y_pred = np.array([0,0,1,1,0,0,1,1])
        from sklearn.metrics import f1_score
        mean, lo, hi = bootstrap_ci(y_true, y_pred,
                                     lambda yt,yp: f1_score(yt,yp,zero_division=0),
                                     n_boot=200)
        assert lo <= mean <= hi
        assert 0.0 <= lo and hi <= 1.0


class TestPerAttackMetrics:
    def test_output_columns(self):
        y_mc   = np.array([0,0,1,1,2,2,3,3])
        y_pred = np.array([0,0,1,0,1,1,0,0])
        label_map = {0:"legit",1:"pm",2:"ghost",3:"vd"}
        df = per_attack_metrics(y_mc, y_pred, label_map)
        assert "Detection Rate %" in df.columns
        assert "FAR %" in df.columns

    def test_legitimate_no_dr(self):
        y_mc   = np.array([0,0,0,1,1])
        y_pred = np.array([0,0,0,1,1])
        label_map = {0:"legit",1:"attack"}
        df = per_attack_metrics(y_mc, y_pred, label_map)
        legit_row = df[df["Attack Type"]=="legit"]
        assert len(legit_row) == 1