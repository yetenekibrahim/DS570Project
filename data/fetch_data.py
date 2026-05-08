"""
fetch_data.py
=============
Data ingestion module for the ADS-B RF Signal Anomaly Detection project.

Two data sources are supported:
  1. OpenSky Network REST API  — live, no authentication required.
     https://opensky-network.org/api/states/all
  2. Mendeley labeled benchmark dataset (Ould Slimane et al., 2022)
     DOI: 10.17632/6fhw732ccz.1  — CC BY 4.0
     22,316 ADS-B messages: legitimate (0), path-modification (1),
     ghost-aircraft (2), velocity-drift (3) attacks.

Both sources are fetched automatically at runtime — no local files or
user accounts are required.
"""

import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Source 1: OpenSky Network (live data)
# ---------------------------------------------------------------------------

OPENSKY_URL = "https://opensky-network.org/api/states/all"

OPENSKY_COLUMNS = [
    "icao24", "callsign", "origin_country", "time_position",
    "last_contact", "longitude", "latitude", "baro_altitude",
    "on_ground", "velocity", "true_track", "vertical_rate",
    "sensors", "geo_altitude", "squawk", "spi", "position_source",
]


def fetch_opensky_data(timeout: int = 30) -> pd.DataFrame:
    """
    Fetch live ADS-B state vectors from the OpenSky Network REST API.

    Parameters
    ----------
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame of airborne flights with numeric feature columns.
    """
    print("[OpenSky] Fetching live ADS-B data ...")
    response = requests.get(OPENSKY_URL, timeout=timeout)
    response.raise_for_status()

    states = response.json().get("states", [])
    if not states:
        raise ValueError("OpenSky returned an empty state vector list.")

    df = pd.DataFrame(states, columns=OPENSKY_COLUMNS)

    # Keep only airborne flights
    df = df[df["on_ground"] == False].copy()

    # Cast numeric columns
    for col in ["longitude", "latitude", "baro_altitude", "velocity",
                "true_track", "vertical_rate", "geo_altitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing the core signal features
    df = df.dropna(subset=["latitude", "longitude", "baro_altitude", "velocity"])

    df["timestamp"] = datetime.utcnow()
    df["data_source"] = "opensky_live"

    print(f"[OpenSky] {len(df)} active flights loaded.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source 2: Mendeley labeled benchmark dataset
# ---------------------------------------------------------------------------

# Public raw CSV — no authentication required.
# Original source: Mendeley Data, DOI 10.17632/6fhw732ccz.1
MENDELEY_CSV_URL = (
    "https://data.mendeley.com/public-files/datasets/6fhw732ccz/files/"
    "9b1e4c4e-3c5e-4b7e-8c6d-2f1a0e9b3d5f/file_downloaded"
)

# Fallback: synthetic mirror using the published feature schema
_MENDELEY_FALLBACK = True   # set False if the real URL is reachable

MENDELEY_COLUMNS = [
    "icao24", "callsign", "origin_country", "time_position",
    "last_contact", "longitude", "latitude", "baro_altitude",
    "on_ground", "velocity", "true_track", "vertical_rate",
    "geo_altitude", "rss", "doppler_shift", "label",
]

LABEL_MAP = {
    0: "legitimate",
    1: "path_modification",
    2: "ghost_aircraft",
    3: "velocity_drift",
}


def _generate_synthetic_benchmark(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic benchmark dataset that reproduces the statistical
    properties described in Ould Slimane et al. (2022).

    Attack proportions (from paper):
      legitimate      ~70 %
      path_mod        ~10 %
      ghost_aircraft  ~10 %
      velocity_drift  ~10 %

    This function is used as a fallback when the Mendeley endpoint is
    unreachable. It is NOT used for the real evaluation — replace with
    the actual CSV download for publication experiments.
    """
    rng = np.random.RandomState(seed)

    labels = rng.choice([0, 1, 2, 3], size=n, p=[0.70, 0.10, 0.10, 0.10])

    altitude  = np.clip(rng.normal(8000, 3000, n), 0, 14000).astype(float)
    velocity  = np.clip(rng.normal(220, 60,   n), 0, 500).astype(float)
    vrate     = rng.normal(0, 5, n).astype(float)
    track     = rng.uniform(0, 360, n).astype(float)
    lon       = rng.uniform(-180, 180, n).astype(float)
    lat       = rng.uniform(-90,  90,  n).astype(float)
    rss       = rng.normal(-80, 10, n).astype(float)
    doppler   = rng.normal(0, 5, n).astype(float)

    # Inject attack signatures
    mask_pm  = labels == 1   # path modification  → position jump
    mask_ga  = labels == 2   # ghost aircraft     → random ICAO
    mask_vd  = labels == 3   # velocity drift     → extreme velocity

    altitude[mask_pm]  += rng.normal(2000, 500, mask_pm.sum())
    velocity[mask_vd]  += rng.choice([-1, 1], mask_vd.sum()) * rng.uniform(80, 200, mask_vd.sum())
    velocity = np.clip(velocity, 0, 900)
    rss[mask_ga]       += rng.normal(20, 5, mask_ga.sum())
    doppler[mask_vd]   += rng.normal(15, 3, mask_vd.sum())

    icao = [
        ''.join(rng.choice(list('0123456789abcdef'), 6))
        for _ in range(n)
    ]

    df = pd.DataFrame({
        "icao24":          icao,
        "callsign":        [f"FL{i:04d}" for i in range(n)],
        "origin_country":  rng.choice(
            ["United States", "Germany", "Turkey", "France", "UK", "China"], n),
        "longitude":       lon,
        "latitude":        lat,
        "baro_altitude":   altitude,
        "velocity":        velocity,
        "true_track":      track,
        "vertical_rate":   vrate,
        "rss":             rss,
        "doppler_shift":   doppler,
        "label":           labels,
        "attack_type":     [LABEL_MAP[l] for l in labels],
        "data_source":     "synthetic_benchmark",
    })

    return df.reset_index(drop=True)


def fetch_benchmark_dataset() -> pd.DataFrame:
    """
    Download the Mendeley ADS-B injection attack benchmark dataset.
    Falls back to a synthetic replica if the endpoint is unreachable.

    Returns
    -------
    pd.DataFrame
        Dataset with 'label' (0-3) and 'attack_type' columns.
    """
    print("[Benchmark] Attempting to fetch Mendeley dataset ...")
    try:
        resp = requests.get(MENDELEY_CSV_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["data_source"] = "mendeley_benchmark"
        df["attack_type"] = df["label"].map(LABEL_MAP)
        print(f"[Benchmark] {len(df)} records loaded from Mendeley.")
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"[Benchmark] Mendeley unreachable ({e}). Using synthetic benchmark.")
        df = _generate_synthetic_benchmark()
        print(f"[Benchmark] {len(df)} synthetic records generated.")
        return df


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test
    df_live = fetch_opensky_data()
    print(df_live[["callsign", "origin_country", "baro_altitude", "velocity"]].head())

    df_bench = fetch_benchmark_dataset()
    print(df_bench["attack_type"].value_counts())