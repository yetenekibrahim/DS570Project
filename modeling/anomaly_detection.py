import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

FEATURE_COLS = ["baro_altitude", "velocity", "altitude_velocity_ratio"]

def prepare_features(df: pd.DataFrame):
    """
    Select and scale features for ML model.
    StandardScaler used because Isolation Forest is distance-sensitive.
    """
    X = df[FEATURE_COLS].dropna().values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_isolation_forest(X: np.ndarray, contamination: float = 0.05):
    """
    Isolation Forest for unsupervised anomaly detection.
    
    Why Isolation Forest?
    - No labeled data needed (unsupervised)
    - Works well with high-dimensional tabular data
    - Robust to outliers by design
    - Faster than density-based methods (e.g. LOF) on large datasets
    
    contamination: expected fraction of anomalies in the dataset
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    return model

def detect_anomalies(model, X: np.ndarray) -> np.ndarray:
    """Returns -1 for anomalies, 1 for normal flights"""
    return model.predict(X)

def get_anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """
    Anomaly scores: more negative = more anomalous.
    Useful for ranking flights by suspiciousness.
    """
    return model.score_samples(X)

def evaluate_model(predictions: np.ndarray) -> dict:
    """Basic evaluation statistics"""
    n_anomalies = int(np.sum(predictions == -1))
    n_normal = int(np.sum(predictions == 1))
    total = len(predictions)
    return {
        "total": total,
        "normal": n_normal,
        "anomalies": n_anomalies,
        "anomaly_rate": round(n_anomalies / total * 100, 2)
    }