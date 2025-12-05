import nflreadpy as nfl
import pandas as pd
import numpy as np
import re

def load_data():
    all_years = list(range(1999, 2025))

    df = nfl.load_pbp(all_years)
    df = df.to_pandas()

    df = df[df["field_goal_attempt"] == 1].copy()

    keep_cols = [
        "kick_distance",
        "field_goal_result",
        "roof",
        "surface",
        "weather",
        "score_differential",
        "half_seconds_remaining",
        "wind",
        "temp",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


def parse_weather(weather_str: str):
    if not isinstance(weather_str, str):
        weather_str = "unknown"
    w_raw = weather_str.strip()
    w_lower = w_raw.lower()

    temp_f = np.nan
    humidity_pct = np.nan
    wind_speed_mph = np.nan
    wind_dir = None

    desc = w_raw
    idx_temp = w_lower.find("temp:")
    if idx_temp != -1:
        desc = w_raw[:idx_temp]
    else:
        desc = w_raw.split(",")[0]
    weather_desc = desc.strip()

    m_temp = re.search(r"temp:\s*([-\d]+)", w_lower)
    if m_temp:
        try:
            temp_f = float(m_temp.group(1))
        except ValueError:
            temp_f = np.nan

    m_hum = re.search(r"humidity:\s*(\d+)", w_lower)
    if m_hum:
        try:
            humidity_pct = float(m_hum.group(1))
        except ValueError:
            humidity_pct = np.nan

    m_wind = re.search(r"wind:\s*([a-z]+)\s+(\d+)\s*mph", w_lower)
    if m_wind:
        wind_dir = m_wind.group(1).upper()
        try:
            wind_speed_mph = float(m_wind.group(2))
        except ValueError:
            wind_speed_mph = np.nan
    else:
        if "wind:" in w_lower and "calm" in w_lower:
            wind_dir = "CALM"
            wind_speed_mph = 0.0

    desc_lower = weather_desc.lower()
    if "rain" in desc_lower or "shower" in desc_lower or "drizzle" in desc_lower:
        weather_type = "rain"
    elif "snow" in desc_lower or "sleet" in desc_lower:
        weather_type = "snow"
    elif "clear" in desc_lower or "sunny" in desc_lower:
        weather_type = "clear"
    elif "cloud" in desc_lower or "overcast" in desc_lower:
        weather_type = "cloudy"
    elif "fog" in desc_lower:
        weather_type = "fog"
    elif weather_str == "unknown":
        weather_type = "unknown"
    else:
        weather_type = "other"

    return pd.Series(
        {
            "weather_desc": weather_desc,
            "weather_type": weather_type,
            "temp_f": temp_f,
            "humidity_pct": humidity_pct,
            "wind_dir": wind_dir,
            "wind_speed_mph": wind_speed_mph,
        }
    )


def clean_data(df: pd.DataFrame | None = None):
    df["fg_made"] = df["field_goal_result"].map(
        {"made": 1, "missed": 0, "blocked": 0}
    )
    df["fg_made"] = df["fg_made"].astype("int")

    df = df.dropna(subset=["kick_distance"])
    df["kick_distance"] = df["kick_distance"].astype(float)

    df["weather_missing"] = df["weather"].isna().astype(int)
    df["weather"] = df["weather"].fillna("unknown").astype(str)

    parsed_weather = df["weather"].apply(parse_weather)
    df = pd.concat([df, parsed_weather], axis=1)

    df["temp_f"] = df["temp_f"].fillna(df["temp"])
    df["wind_speed_mph"] = df["wind_speed_mph"].fillna(df["wind"])

    df["is_indoor"] = df["roof"].isin(["closed", "indoor", "dome"]).astype(int)

    model_cols = [
        "kick_distance",
        "roof",
        "surface",
        "half_seconds_remaining",
        "score_differential",
        "wind_speed_mph",
        "wind_dir",
        "humidity_pct",
        "weather_type",
        "temp_f",
        "fg_made",
    ]

    df_model = df[model_cols].copy()
    df_model.to_csv("../data/fg_data.csv", index=False)


def main():
    data = load_data()
    clean_data(data)


if __name__ == "__main__":
    main()