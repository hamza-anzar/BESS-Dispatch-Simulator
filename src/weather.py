# weather.py
# -------------------------------------------------------------
# Weather data + renewable generation profiles for hybrid plants
#   - Download hourly sun + wind data from NASA POWER (free, no API key)
#   - Convert sunlight -> PV power and wind speed -> wind turbine power
#   - Read user-uploaded weather / generation CSV files
#
# NASA POWER delay (checked Oct 2026):
#   wind speed  ~ 1 week behind today
#   solar (GHI) ~ 3 months behind today
# Missing hours are filled with a "typical year" value (average of the same
# calendar day + hour in the years that DO have data). Column "filled" = True
# marks those hours, so results can say which days used estimated weather.
# -------------------------------------------------------------
import numpy as np
import pandas as pd
import requests
import config

NASA_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_PARAMS = {
    "ALLSKY_SFC_SW_DWN": "ghi_w_m2",        # sunlight on a flat surface (W/m2)
    "WS50M": "wind_50m_m_s",                 # wind speed at 50 m height (m/s)
    "T2M": "temp_c",                         # air temperature at 2 m (deg C)
}


def _cache_file(lat, lon):
    """One cache CSV per location, e.g. data/weather/nasa_51.16_10.45.csv"""
    return config.DATA_WEATHER / f"nasa_{lat:.2f}_{lon:.2f}.csv"


# ============================================================
# 1. Download from NASA POWER
# ============================================================
def download_nasa(lat, lon, start, end):
    """Download hourly NASA POWER data between two dates (texts 'YYYY-MM-DD', end inclusive).
    Returns a table with a UTC index; missing values (-999) become NaN."""
    parts = []
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    # NASA allows long requests, but one year per request is safer
    for year in range(first.year, last.year + 1):
        a = max(first, pd.Timestamp(f"{year}-01-01"))
        b = min(last, pd.Timestamp(f"{year}-12-31"))
        params = {
            "parameters": ",".join(NASA_PARAMS),
            "community": "RE",
            "latitude": lat,
            "longitude": lon,
            "start": a.strftime("%Y%m%d"),
            "end": b.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
        }
        print(f"NASA POWER: {a.date()} -> {b.date()} at ({lat}, {lon}) ...")
        r = requests.get(NASA_URL, params=params, timeout=120)
        r.raise_for_status()
        data = r.json()["properties"]["parameter"]
        df = pd.DataFrame(data)                               # rows = "YYYYMMDDHH"
        df.index = pd.to_datetime(df.index, format="%Y%m%d%H").tz_localize("UTC")
        parts.append(df)

    w = pd.concat(parts).rename(columns=NASA_PARAMS)
    w = w[~w.index.duplicated(keep="last")].sort_index()
    return w.replace(-999.0, np.nan)


def get_weather(lat=config.LATITUDE, lon=config.LONGITUDE,
                start=config.START, end=None, refresh=False):
    """Weather for a location, using a local cache so NASA is only asked once.
    The cache is topped up automatically when the wind data is more than 10 days old;
    refresh=True forces a top-up (solar data arrives ~3 months late)."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    path = _cache_file(lat, lon)

    if path.exists():
        w = pd.read_csv(path, index_col=0)
        w.index = pd.to_datetime(w.index, utc=True)
        # (1 day tolerance: the first Berlin hour is 23:00 UTC of the previous day)
        have_start = w.index.min() <= pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=1)
        last_full = w.dropna().index.max()                    # last hour with ALL values
        last_wind = w["wind_50m_m_s"].dropna().index.max()    # wind arrives ~1 week late
        stale = last_wind < pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=10)
        if have_start and not refresh and not stale:
            return w
        if have_start:
            # Download only from the last complete hour onwards and merge
            new_from = (last_full - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new = download_nasa(lat, lon, new_from, end)
            w = pd.concat([w, new])
            w = w[~w.index.duplicated(keep="last")].sort_index()
            w.to_csv(path, index_label="timestamp_utc")
            return w

    w = download_nasa(lat, lon, start, end)
    w.to_csv(path, index_label="timestamp_utc")
    return w


# ============================================================
# 2. Align weather to the price timestamps + fill gaps
# ============================================================
def align_weather(weather, index):
    """Put weather on the same hours as the prices (index = Berlin timestamps).
    Missing hours are filled with the typical-year value for that day + hour."""
    idx_utc = index.tz_convert("UTC")
    weather = weather[[c for c in weather.columns if not c.endswith("filled")]]
    w = weather.reindex(idx_utc)
    filled = w.isna()                                          # True where we must estimate

    # Typical year: average per (month, day, hour) over all years with data
    keys = [weather.index.month, weather.index.day, weather.index.hour]
    clim = weather.groupby(keys).mean()
    lookup = pd.MultiIndex.from_arrays([idx_utc.month, idx_utc.day, idx_utc.hour])
    clim_vals = clim.reindex(lookup)
    clim_vals.index = idx_utc
    w = w.fillna(clim_vals)

    # Anything still missing (e.g. 29 Feb) -> straight line between neighbours
    w = w.interpolate(limit_direction="both")
    for c in filled.columns:                                   # e.g. "ghi_w_m2_filled"
        w[f"{c}_filled"] = filled[c].values
    w.index = index
    return w


# ============================================================
# 3. Weather -> power (MW)
# ============================================================
def pv_power(ghi_w_m2, peak_mw=config.PV_PEAK_MW, pr=config.PV_PR):
    """PV output (MW) = peak * (sunlight / 1000 W/m2) * performance ratio."""
    ghi = np.clip(np.asarray(ghi_w_m2, dtype=float), 0, None)
    return np.clip(peak_mw * ghi / 1000.0 * pr, 0, peak_mw)


def wind_power(wind_50m_m_s, rated_mw=config.WIND_RATED_MW, hub_height=config.WIND_HUB_HEIGHT_M,
               cut_in=config.WIND_CUT_IN, rated_speed=config.WIND_RATED_SPEED,
               cut_out=config.WIND_CUT_OUT, alpha=config.WIND_SHEAR_ALPHA):
    """Wind output (MW) from a simple turbine power curve.
    1. Scale wind from 50 m to hub height: v_hub = v_50 * (hub / 50) ** alpha
    2. Below cut-in or above cut-out -> 0; between cut-in and rated speed the
       power grows with v**3; from rated speed to cut-out -> full power."""
    v = np.clip(np.asarray(wind_50m_m_s, dtype=float), 0, None) * (hub_height / 50.0) ** alpha
    p = np.zeros_like(v)
    ramp = (v >= cut_in) & (v < rated_speed)
    p[ramp] = rated_mw * (v[ramp] ** 3 - cut_in ** 3) / (rated_speed ** 3 - cut_in ** 3)
    p[(v >= rated_speed) & (v <= cut_out)] = rated_mw
    return p


def generation_profile(index, mode=config.ASSET_MODE, weather=None, **kw):
    """Renewable generation in MW for every price hour (index = Berlin timestamps).
    mode 'grid' -> zeros.  'solar' / 'wind' -> from NASA weather (or the 'weather' table given).
    Extra keywords go to pv_power / wind_power (e.g. peak_mw=5).
    Returns a table with columns: gen_mw, filled (+ the weather columns)."""
    if mode == "grid":
        return pd.DataFrame({"gen_mw": 0.0, "filled": False}, index=index)

    if weather is None:
        start = index.min().tz_convert("UTC").strftime("%Y-%m-%d")
        w = get_weather(start=min(start, config.START))
    else:
        w = weather
    w = align_weather(w, index)

    if mode == "solar":
        w["gen_mw"] = pv_power(w["ghi_w_m2"], **kw)
        w["filled"] = w["ghi_w_m2_filled"]
    elif mode == "wind":
        w["gen_mw"] = wind_power(w["wind_50m_m_s"], **kw)
        w["filled"] = w["wind_50m_m_s_filled"]
    else:
        raise ValueError(f"Unknown mode: {mode} (use 'grid', 'solar' or 'wind')")
    return w


# ============================================================
# 4. User-uploaded files (for the Streamlit app)
# ============================================================
def weather_from_csv(file):
    """Read an uploaded hourly file. First column (or 'timestamp') = time.
    Accepted value columns (any of them):
      generation_mw  -> used directly as the park's output
      ghi_w_m2       -> sunlight, converted with pv_power()
      wind_50m_m_s   -> wind speed at 50 m, converted with wind_power()
      wind_hub_m_s   -> wind speed already at hub height
    Times without a timezone are treated as Berlin local time. Returns a UTC-indexed table."""
    df = pd.read_csv(file)
    time_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    try:
        times = pd.to_datetime(df[time_col])
    except (ValueError, TypeError):
        times = pd.to_datetime(df[time_col], utc=True)
    if times.dt.tz is None:
        times = times.dt.tz_localize(config.LOCAL_TZ, ambiguous="NaT",
                                     nonexistent="shift_forward")
    df.index = pd.DatetimeIndex(times).tz_convert("UTC")
    df = df[df.index.notna()].drop(columns=[time_col])

    if "wind_hub_m_s" in df.columns and "wind_50m_m_s" not in df.columns:
        # convert hub-height wind back to 50 m so wind_power() can scale it up again
        df["wind_50m_m_s"] = df["wind_hub_m_s"] / (config.WIND_HUB_HEIGHT_M / 50.0) ** config.WIND_SHEAR_ALPHA
    known = [c for c in ["generation_mw", "ghi_w_m2", "wind_50m_m_s", "temp_c"] if c in df.columns]
    if not known:
        raise ValueError("Upload needs a column: generation_mw, ghi_w_m2, wind_50m_m_s or wind_hub_m_s")
    df = df[known].apply(pd.to_numeric, errors="coerce")
    return df.resample("h").mean()


def generation_from_upload(upload, index, mode, **kw):
    """Like generation_profile(), but from an uploaded table (see weather_from_csv)."""
    w = align_weather(upload, index)
    if "generation_mw" in w.columns:
        w["gen_mw"] = w["generation_mw"].clip(lower=0)
        w["filled"] = w["generation_mw_filled"]
    elif mode == "solar" and "ghi_w_m2" in w.columns:
        w["gen_mw"] = pv_power(w["ghi_w_m2"], **kw)
        w["filled"] = w["ghi_w_m2_filled"]
    elif mode == "wind" and "wind_50m_m_s" in w.columns:
        w["gen_mw"] = wind_power(w["wind_50m_m_s"], **kw)
        w["filled"] = w["wind_50m_m_s_filled"]
    else:
        raise ValueError(f"The uploaded file has no column usable for mode '{mode}'")
    return w


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_loader import load_prices

    prices = load_prices()
    w = get_weather()
    print(w.describe().round(2))
    print("Last hour with sunlight data:", w["ghi_w_m2"].dropna().index.max())
    print("Last hour with wind data    :", w["wind_50m_m_s"].dropna().index.max())

    solar = generation_profile(prices.index, mode="solar", weather=w)
    wind = generation_profile(prices.index, mode="wind", weather=w)
    print(f"PV  {config.PV_PEAK_MW} MWp : {solar['gen_mw'].sum() * config.DT:,.0f} MWh "
          f"({solar['filled'].mean():.1%} of hours estimated)")
    print(f"Wind {config.WIND_RATED_MW} MW : {wind['gen_mw'].sum() * config.DT:,.0f} MWh "
          f"({wind['filled'].mean():.1%} of hours estimated)")

    # Quick look at one summer week
    week = slice("2024-06-10", "2024-06-16")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(solar.loc[week, "gen_mw"], label="PV (MW)")
    ax.plot(wind.loc[week, "gen_mw"], label="Wind (MW)")
    ax.set_ylabel("MW")
    ax.set_title("Renewable generation from NASA POWER weather")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "14_generation_week.png", dpi=150)
    plt.show()
