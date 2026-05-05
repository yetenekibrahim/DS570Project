"""
attack_injection.py
===================
Physics-constrained ADS-B attack injection for benchmark generation.

Attack Types & Spectral Signatures
------------------------------------
1. PATH MODIFICATION (label=1)
   Position fields altered. Velocity unchanged but neighbours now include
   flights at very different altitudes → local velocity signal becomes
   heterogeneous → moderate entropy increase.

2. GHOST AIRCRAFT (label=2)
   Fabricated state vectors with velocity drawn Uniform(10,450) instead
   of Normal(220,55). In the local neighbourhood, ghost aircraft cluster
   with other ghosts (similar random ICAO prefix 'ff...') producing
   high spectral flatness and entropy.

3. VELOCITY DRIFT (label=3)
   Velocity step-change ±[120,280] m/s — well beyond max aircraft
   acceleration (0.5 m/s² × 5 s = 2.5 m/s). Neighbours of a
   velocity-drift aircraft are other fast/slow outliers → local
   velocity signal has extreme range → very high entropy, low BER.

4. REPLAY (label=4)
   Position displaced but velocity from a real record — velocity looks
   normal individually but its spatial neighbourhood is inconsistent.
   Spectral signature: slightly elevated entropy, shifted centroid.

Physical Bounds
---------------
All generated values are clipped to ADS-B field ranges:
  baro_altitude ∈ [0, 15000] m
  velocity      ∈ [0, 900]   m/s
  rss           ∈ [-120, -20] dBm
"""

import numpy as np
import pandas as pd

ATTACK_LABELS = {
    0: "legitimate",
    1: "path_modification",
    2: "ghost_aircraft",
    3: "velocity_drift",
    4: "replay",
}

ALT_MIN, ALT_MAX  =   0.0, 15000.0
VEL_MIN, VEL_MAX  =   0.0,   900.0
RSS_MIN, RSS_MAX  = -120.0,   -20.0


def _icao(rng, n, prefix=""):
    chars = list("0123456789abcdef")
    length = 6 - len(prefix)
    return [prefix + "".join(rng.choice(chars, length)) for _ in range(n)]


def generate_legitimate_flights(n: int = 3000,
                                 rng: np.random.RandomState = None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.RandomState(42)

    altitude = np.clip(rng.normal(8000, 2500, n), 300,  14000).astype(float)
    velocity = np.clip(rng.normal(220,  55,   n),  60,    500).astype(float)
    vrate    = np.clip(rng.normal(0,    4,    n), -80,    80).astype(float)
    rss      = np.clip(rng.normal(-85,  8,    n), RSS_MIN, RSS_MAX).astype(float)
    doppler  = rng.normal(0, 3, n).astype(float)

    countries = ["United States","Germany","Turkey","France","UK",
                 "China","Japan","India","Brazil","Australia"]
    return pd.DataFrame({
        "icao24":         _icao(rng, n),
        "callsign":       [f"FL{i:05d}" for i in range(n)],
        "origin_country": rng.choice(countries, n),
        "longitude":      rng.uniform(-180, 180, n),
        "latitude":       rng.uniform(-60,   75, n),
        "baro_altitude":  altitude,
        "velocity":       velocity,
        "true_track":     rng.uniform(0, 360, n),
        "vertical_rate":  vrate,
        "rss":            rss,
        "doppler_shift":  doppler,
        "label":          0,
        "attack_type":    "legitimate",
    })


def inject_path_modification(df_legit, n_attacks, rng):
    """
    Altitude displaced by ±[3000,6000] m — large enough to push the
    flight into a completely different neighbourhood in feature space.
    """
    idx    = rng.choice(len(df_legit), n_attacks, replace=False)
    df_att = df_legit.iloc[idx].copy().reset_index(drop=True)

    sign      = rng.choice([-1,1], n_attacks)
    delta_alt = sign * rng.uniform(3000, 6000, n_attacks)
    df_att["baro_altitude"] = np.clip(df_att["baro_altitude"] + delta_alt,
                                       ALT_MIN, ALT_MAX)
    df_att["latitude"]  += rng.normal(0, 2.5, n_attacks)
    df_att["longitude"] += rng.normal(0, 2.5, n_attacks)
    df_att["rss"]        = np.clip(df_att["rss"] - rng.uniform(8,18,n_attacks),
                                    RSS_MIN, RSS_MAX)
    df_att["label"]       = 1
    df_att["attack_type"] = "path_modification"
    return df_att


def inject_ghost_aircraft(n_attacks, rng):
    """
    Fully fabricated flights with velocity drawn from Uniform(10,450)
    — much broader than the legitimate Normal(220,55).
    ICAO prefixed 'ff' to cluster ghosts together in KNN space.
    """
    altitude = rng.uniform(500, 12000, n_attacks).astype(float)
    velocity = rng.uniform(10,  450,   n_attacks).astype(float)   # broad uniform
    rss      = np.clip(rng.normal(-62, 7, n_attacks), RSS_MIN, RSS_MAX).astype(float)
    doppler  = rng.normal(0, 0.4, n_attacks).astype(float)

    return pd.DataFrame({
        "icao24":         _icao(rng, n_attacks, prefix="ff"),
        "callsign":       [f"GH{i:05d}" for i in range(n_attacks)],
        "origin_country": "Unknown",
        "longitude":      rng.uniform(-180, 180, n_attacks),
        "latitude":       rng.uniform(-60,   75, n_attacks),
        "baro_altitude":  altitude,
        "velocity":       velocity,
        "true_track":     rng.uniform(0, 360, n_attacks),
        "vertical_rate":  rng.uniform(-60, 60, n_attacks),
        "rss":            rss,
        "doppler_shift":  doppler,
        "label":          2,
        "attack_type":    "ghost_aircraft",
    })


def inject_velocity_drift(df_legit, n_attacks, rng):
    """
    Velocity step-change ±[120,280] m/s — physically impossible for
    commercial aircraft (max aerodynamic acceleration ≈ 2.5 m/s per
    5-second SSR cycle). Doppler anomaly proportional.
    """
    idx    = rng.choice(len(df_legit), n_attacks, replace=False)
    df_att = df_legit.iloc[idx].copy().reset_index(drop=True)

    sign    = rng.choice([-1,1], n_attacks)
    delta_v = sign * rng.uniform(120, 280, n_attacks)
    df_att["velocity"]      = np.clip(df_att["velocity"] + delta_v, VEL_MIN, VEL_MAX)
    df_att["doppler_shift"] += sign * rng.uniform(15, 35, n_attacks)
    df_att["label"]          = 3
    df_att["attack_type"]    = "velocity_drift"
    return df_att


def inject_replay_attack(df_legit, n_attacks, rng):
    """
    State vectors copied from a different record and displaced in position.
    Velocity is internally consistent but spatially displaced.
    """
    idx1 = rng.choice(len(df_legit), n_attacks, replace=False)
    idx2 = rng.choice(len(df_legit), n_attacks, replace=False)
    df_att = df_legit.iloc[idx1].copy().reset_index(drop=True)
    src    = df_legit.iloc[idx2].reset_index(drop=True)

    df_att["latitude"]      = src["latitude"].values
    df_att["longitude"]     = src["longitude"].values
    df_att["baro_altitude"] = src["baro_altitude"].values
    df_att["rss"]           = np.clip(df_att["rss"] + rng.uniform(5,14,n_attacks),
                                       RSS_MIN, RSS_MAX)
    df_att["doppler_shift"] += rng.normal(0, 5, n_attacks)
    df_att["label"]          = 4
    df_att["attack_type"]    = "replay"
    return df_att


def build_benchmark_dataset(n_legit=3000, n_pm=350, n_ghost=350,
                              n_vd=350, n_replay=300, seed=42) -> pd.DataFrame:
    """
    Build the full labeled benchmark dataset.

    Class proportions approximate the Mendeley ADS-B injection dataset
    (Ould Slimane et al., 2022, DOI: 10.17632/6fhw732ccz.1).
    """
    rng = np.random.RandomState(seed)
    df_legit  = generate_legitimate_flights(n_legit, rng)
    df_pm     = inject_path_modification(df_legit, n_pm,    rng)
    df_ghost  = inject_ghost_aircraft(n_ghost, rng)
    df_vd     = inject_velocity_drift(df_legit, n_vd, rng)
    df_replay = inject_replay_attack(df_legit,  n_replay, rng)

    df = pd.concat([df_legit, df_pm, df_ghost, df_vd, df_replay],
                   ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[Dataset] Built benchmark: {len(df)} records")
    print(df["attack_type"].value_counts().to_string())
    return df


if __name__ == "__main__":
    df = build_benchmark_dataset()
    print("\nSample:\n", df[["attack_type","baro_altitude","velocity","rss"]].head(8))