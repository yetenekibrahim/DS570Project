# SpectralIF — ADS-B RF Signal Anomaly Detection

> **DS570 Data Science Final Project**  
> Ibrahim Yetenek · Özyeğin University · Spring 2026  
> Target venue: IEEE Access / MDPI Sensors (Q2)

---

## Overview

**SpectralIF** is a novel, label-free, explainable anomaly detection
framework for ADS-B aviation signals. It trains **two separate Isolation
Forests** — one on kinematic features, one on Local Spectral Neighborhood
Features (LSNF) derived from per-flight FFT analysis — and fuses their
scores with a validation-optimised weight α:

$$\mathcal{A}(x) = \alpha\,\hat{s}_{base}(x) + (1-\alpha)\,\hat{s}_{spec}(x)$$

α* is found by grid search on an inner validation fold, ensuring no
leakage between train and test splits.

---

## Key Results (5-Fold Cross-Validation, ρ=0.20)

| Method | F1 ± std | AUC-ROC | Latency |
|--------|----------|---------|---------|
| Z-Score | 0.341 ± 0.034 | 0.876 | ~0 ms |
| LOF | 0.379 ± 0.028 | 0.639 | 12 ms |
| IsoForest (base) | 0.703 ± 0.019 | 0.879 | 21 ms |
| **SpectralIF** | **0.714 ± 0.023** | **0.876** | **~30 ms** |

Velocity-drift detection rate: **98.5%** · Inference: **~280× faster than xLSTM-IDS**

---

## Project Structure

```
DS570Project/
├── Dockerfile
├── README.md
├── requirements.txt
├── data/
│   ├── fetch_data.py          # OpenSky Network API (live, no auth)
│   └── attack_injection.py    # Physics-constrained benchmark dataset
├── processing/
│   └── signal_processing.py  # Vectorized LSNF + LSNFTransformer
├── modeling/
│   ├── spectralif.py          # SpectralIF class (two-detector fusion)
│   ├── anomaly_detection.py   # Supporting IF utilities + SHAP
│   ├── baseline.py            # Z-Score and LOF baselines
│   └── evaluation.py          # Bootstrap CI, ablation, CV comparison
├── app/
│   └── dashboard.py          # 10-section Streamlit dashboard
└── tests/
    ├── test_signal_processing.py
    └── test_anomaly_detection.py
```

---

## How to Run

### Option 1 — Docker (recommended)

```bash
# Build
docker build -t spectralif .

# Run
docker run -p 8501:8501 spectralif
```

Open **http://localhost:8501** in your browser.  
The app fetches live ADS-B data automatically — no files or accounts needed.

### Option 2 — Local

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

### Run Tests

```bash
python3 -m pytest tests/ -v
```

---

## Dashboard Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Live Data | OpenSky KPIs, altitude/velocity histograms |
| 2 | Signal Processing | FFT/PSD plot, 6 spectral features |
| 3 | SpectralIF Detection | Anomaly map, score histogram, hybrid score curve |
| 4 | Flight Map | Interactive Plotly map (anomalies in red) |
| 5 | Benchmark | Per-attack DR, ROC/PR curves, comparison table |
| 6 | Ablation Study | Marginal contribution of spectral features |
| 7 | Sensitivity | F1/AUC vs contamination ρ |
| 8 | SHAP | Feature importance explainability |
| 9 | Anomalous Flights | Ranked table of flagged flights |
| 10 | Paper Summary | Contributions & limitations |

---

## Data Sources

| Source | Description | License |
|--------|-------------|---------|
| [OpenSky Network](https://opensky-network.org/) | Live ADS-B state vectors | CC BY 4.0 |
| Physics-constrained synthetic benchmark | 4,350 records, 4 attack types | Generated |
| [Mendeley Dataset](https://data.mendeley.com/datasets/6fhw732ccz/1) | Labeled ADS-B injection attacks | CC BY 4.0 |

---

## Novel Contributions

1. **Two-detector fusion** — separate feature spaces prevent kinematic features from diluting spectral signal.
2. **Validation-optimised α** — grid search on inner fold, no test leakage.
3. **Vectorized LSNF** — batch FFT achieves 116× speedup (0.66 ms / 1,000 samples).
4. **Label-free** — no labelled attacks needed at training time.
5. **SHAP explainability** — addresses black-box gap in deep-learning detectors.

---

## Limitations

- Synthetic benchmark approximates real adversarial hardware; SDR testbed validation is future work.
- Replay attack DR ~14% — velocity fingerprint is preserved, limiting spectral discrimination.
- Contamination ρ must be set by operator; sensitivity analysis shows stability for ρ ∈ [0.15, 0.25].

---

## References

- Liu et al. (2008). *Isolation Forest*. ICDM.
- Lundberg & Lee (2017). *SHAP*. NeurIPS.
- Ould Slimane et al. (2022). Mendeley Data. DOI: 10.17632/6fhw732ccz.1
- Ferrag et al. (2025). *xLSTM-IDS*. IEEE Access.
- Schäfer et al. (2014). *OpenSky*. IPSN.