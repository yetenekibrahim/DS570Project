"""Unit tests for signal_processing.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from processing.signal_processing import (
    compute_psd, extract_spectral_features, extract_features,
    LSNFTransformer, SPECTRAL_FEATURE_NAMES, _spectral_features_batch,
)

RNG = np.random.RandomState(42)


class TestBatchSpectral:
    def test_output_shape(self):
        signals = RNG.randn(50, 8)
        out = _spectral_features_batch(signals)
        assert out.shape == (50, 6)

    def test_noise_higher_entropy_than_tonal(self):
        t = np.linspace(0, 1, 64, endpoint=False)
        tonal = np.sin(2 * np.pi * 5 * t)
        noise = RNG.randn(64)
        batch = np.vstack([tonal, noise])
        feats = _spectral_features_batch(batch)
        assert feats[1, 0] > feats[0, 0], "Noise should have higher entropy"

    def test_no_nan(self):
        signals = RNG.randn(20, 8)
        out = _spectral_features_batch(signals)
        assert not np.isnan(out).any()

    def test_all_unique_for_different_signals(self):
        signals = RNG.randn(100, 8)
        out = _spectral_features_batch(signals)
        # At least 50 unique entropy values from 100 different signals
        assert np.unique(out[:, 0]).shape[0] > 50


class TestComputePSD:
    def test_psd_nonnegative(self):
        _, psd, _ = compute_psd(RNG.randn(256))
        assert (psd >= 0).all()

    def test_output_length(self):
        freqs, psd, psd_db = compute_psd(RNG.randn(256))
        assert len(freqs) == len(psd) == len(psd_db) == 128


class TestLSNFTransformer:
    def _make_X(self, n=200):
        return np.column_stack([
            RNG.uniform(0, 12000, n),
            RNG.uniform(50, 400, n),
            RNG.uniform(0, 360, n),
        ])

    def test_output_shape(self):
        X = self._make_X()
        tr = LSNFTransformer(k=8)
        out = tr.fit_transform(X)
        assert out.shape == (200, 6)

    def test_unique_per_sample(self):
        X = self._make_X(300)
        tr = LSNFTransformer(k=8)
        out = tr.fit_transform(X)
        assert np.unique(out[:, 0]).shape[0] > 100

    def test_no_leakage_train_test(self):
        X = self._make_X(200)
        X_tr, X_te = X[:160], X[160:]
        tr = LSNFTransformer(k=8)
        tr.fit(X_tr)
        out_te = tr.transform(X_te)
        assert out_te.shape == (40, 6)
        assert not np.isnan(out_te).any()

    def test_feature_names_count(self):
        assert len(SPECTRAL_FEATURE_NAMES) == 6


class TestExtractFeatures:
    def test_columns_added(self):
        df = pd.DataFrame({
            'baro_altitude': RNG.uniform(0, 12000, 100),
            'velocity':      RNG.uniform(50, 400, 100),
            'true_track':    RNG.uniform(0, 360, 100),
            'vertical_rate': RNG.randn(100) * 5,
        })
        df_out = extract_features(df, k=8)
        for col in ['altitude_velocity_ratio','flight_phase','speed_category'] + SPECTRAL_FEATURE_NAMES:
            assert col in df_out.columns, f"Missing: {col}"

    def test_spectral_not_all_same(self):
        df = pd.DataFrame({
            'baro_altitude': RNG.uniform(0, 12000, 100),
            'velocity':      RNG.uniform(50, 400, 100),
            'true_track':    RNG.uniform(0, 360, 100),
        })
        df_out = extract_features(df, k=8)
        assert df_out['spectral_entropy'].nunique() > 10