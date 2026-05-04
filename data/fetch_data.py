import requests
import pandas as pd
from datetime import datetime

def fetch_opensky_data():
    """Fetch live ADS-B flight data from OpenSky Network (no auth required)"""
    url = "https://opensky-network.org/api/states/all"
    
    print("Fetching data from OpenSky Network...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    states = data.get("states", [])
    
    columns = [
        "icao24", "callsign", "origin_country", "time_position",
        "last_contact", "longitude", "latitude", "baro_altitude",
        "on_ground", "velocity", "true_track", "vertical_rate",
        "sensors", "geo_altitude", "squawk", "spi", "position_source"
    ]
    
    df = pd.DataFrame(states, columns=columns)
    df = df.dropna(subset=["latitude", "longitude", "baro_altitude", "velocity"])
    df["baro_altitude"] = df["baro_altitude"].astype(float)
    df["velocity"] = df["velocity"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["latitude"] = df["latitude"].astype(float)
    df["timestamp"] = datetime.utcnow()
    
    # Remove ground traffic
    df = df[df["on_ground"] == False]
    
    print(f"Fetched {len(df)} active flights.")
    return df

if __name__ == "__main__":
    df = fetch_opensky_data()
    print(df[["callsign", "origin_country", "baro_altitude", "velocity"]].head(10))