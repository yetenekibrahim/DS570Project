import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_data import fetch_opensky_data
from processing.signal_processing import compute_psd, extract_features, compute_spectral_entropy
from modeling.anomaly_detection import prepare_features, train_isolation_forest, detect_anomalies, get_anomaly_scores, evaluate_model

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ADS-B RF Anomaly Detector",
    page_icon="📡",
    layout="wide"
)

st.title("📡 ADS-B RF Signal Anomaly Detection")
st.markdown("**Real-time flight signal analysis using OpenSky Network** | DS570 Final Project — Ibrahim Yetenek")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")
contamination = st.sidebar.slider("Expected Anomaly Rate (%)", 1, 20, 5) / 100
n_bins = st.sidebar.slider("Histogram Bins", 10, 100, 40)
st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This dashboard fetches live ADS-B transponder data from the 
[OpenSky Network](https://opensky-network.org/) and applies 
signal processing + unsupervised ML to detect anomalous flights.
""")

# ── Data fetch ────────────────────────────────────────────────
with st.spinner("📡 Fetching live flight data from OpenSky Network..."):
    try:
        df_raw = fetch_opensky_data()
        df = extract_features(df_raw)
        st.success(f"✅ {len(df)} active flights loaded from OpenSky Network")
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        st.stop()

# ── KPI Row ───────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Flights", len(df))
k2.metric("Countries", df["origin_country"].nunique())
k3.metric("Avg Altitude (m)", f"{df['baro_altitude'].mean():.0f}")
k4.metric("Avg Velocity (m/s)", f"{df['velocity'].mean():.0f}")
st.divider()

# ── EDA Section ───────────────────────────────────────────────
st.subheader("📊 Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Altitude Distribution**")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df["baro_altitude"], bins=n_bins, color="#4C9BE8", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Barometric Altitude (m)")
    ax.set_ylabel("Number of Flights")
    ax.set_title("Flight Altitude Distribution")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

with col2:
    st.markdown("**Velocity Distribution**")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.hist(df["velocity"], bins=n_bins, color="#E8824C", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Velocity (m/s)")
    ax2.set_ylabel("Number of Flights")
    ax2.set_title("Flight Velocity Distribution")
    ax2.grid(axis="y", alpha=0.3)
    st.pyplot(fig2)

# ── Signal Processing Section ─────────────────────────────────
st.divider()
st.subheader("🔊 Signal Processing — Power Spectral Density")
st.markdown("""
We treat the sorted velocity measurements as a discrete-time signal and compute its 
**Power Spectral Density (PSD)** using the FFT periodogram method. 
Low-frequency components represent slow bulk trends; high-frequency components indicate rapid fluctuations.
""")

velocities = df["velocity"].dropna().values
freqs, psd, psd_db = compute_psd(velocities)
entropy = compute_spectral_entropy(velocities)

col3, col4 = st.columns([3, 1])
with col3:
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(freqs, psd_db, color="#2ecc71", linewidth=1.2)
    ax3.fill_between(freqs, psd_db, alpha=0.2, color="#2ecc71")
    ax3.set_xlabel("Normalized Frequency (cycles/sample)")
    ax3.set_ylabel("Power (dB)")
    ax3.set_title("Power Spectral Density of Velocity Signal")
    ax3.grid(alpha=0.3)
    st.pyplot(fig3)
with col4:
    st.metric("Spectral Entropy", f"{entropy:.3f}")
    st.markdown("Higher entropy → more noise-like signal (diverse flight speeds)")

# ── ML Section ────────────────────────────────────────────────
st.divider()
st.subheader("🚨 Anomaly Detection — Isolation Forest")
st.markdown("""
**Isolation Forest** isolates observations by randomly selecting a feature and split value. 
Anomalies require fewer splits to isolate → shorter path length → lower anomaly score.
""")

df_model = df[["baro_altitude", "velocity", "altitude_velocity_ratio",
               "callsign", "origin_country", "flight_phase"]].dropna().copy()

X_scaled, scaler = prepare_features(df_model)
model = train_isolation_forest(X_scaled, contamination=contamination)
predictions = detect_anomalies(model, X_scaled)
scores = get_anomaly_scores(model, X_scaled)
stats = evaluate_model(predictions)

df_model["anomaly"] = predictions
df_model["anomaly_score"] = scores
df_model["label"] = df_model["anomaly"].map({1: "Normal", -1: "⚠️ Anomaly"})

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Normal Flights", stats["normal"])
m2.metric("⚠️ Anomalous Flights", stats["anomalies"])
m3.metric("Anomaly Rate", f"{stats['anomaly_rate']}%")

col5, col6 = st.columns(2)

with col5:
    st.markdown("**Altitude vs Velocity — Anomaly Map**")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    normal = df_model[df_model["anomaly"] == 1]
    anomalous = df_model[df_model["anomaly"] == -1]
    ax4.scatter(normal["velocity"], normal["baro_altitude"],
                c="#4C9BE8", alpha=0.3, s=4, label="Normal")
    ax4.scatter(anomalous["velocity"], anomalous["baro_altitude"],
                c="red", alpha=0.7, s=12, label="Anomaly")
    ax4.set_xlabel("Velocity (m/s)")
    ax4.set_ylabel("Altitude (m)")
    ax4.set_title("Flight Anomalies")
    ax4.legend()
    ax4.grid(alpha=0.3)
    st.pyplot(fig4)

with col6:
    st.markdown("**Anomaly Score Distribution**")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.hist(scores, bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
    ax5.axvline(x=model.threshold_, color="red", linestyle="--", label="Decision boundary")
    ax5.set_xlabel("Anomaly Score")
    ax5.set_ylabel("Count")
    ax5.set_title("Isolation Forest Score Distribution")
    ax5.legend()
    ax5.grid(alpha=0.3)
    st.pyplot(fig5)

# ── Anomaly Table ─────────────────────────────────────────────
st.divider()
st.subheader("📋 Most Anomalous Flights")
top_anomalies = df_model[df_model["anomaly"] == -1].sort_values("anomaly_score").head(20)
st.dataframe(
    top_anomalies[["callsign", "origin_country", "baro_altitude", "velocity",
                   "altitude_velocity_ratio", "flight_phase", "anomaly_score"]],
    use_container_width=True
)