# data_loader.py
# -------------------------------------------------------------
# STEP 1: Download, clean, save and UPDATE day-ahead prices.
#   - First run : downloads everything from config.START to config.END
#   - Later runs: downloads only the days that are missing
# -------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
import config

# File locations (built from the folder paths in config.py)
RAW_FILE = config.DATA_RAW / "prices_raw.csv"
CLEAN_FILE = config.DATA_PROCESSED / "prices_clean.csv"


def download_prices(start=config.START, end=config.END, country=config.COUNTRY_CODE):
    """Download day-ahead prices (EUR/MWh) from ENTSO-E.
    start / end are date texts like "2024-01-01" (end is exclusive)."""
    # 1. Read the API key from the text file
    api_key = config.read_api_key()

    # 2. Connect to ENTSO-E
    client = EntsoePandasClient(api_key=api_key)

    # 3. Timezone-aware start and end
    start_ts = pd.Timestamp(start, tz=config.LOCAL_TZ)
    end_ts = pd.Timestamp(end, tz=config.LOCAL_TZ)

    # 4. Download (entsoe-py splits long periods into smaller requests itself)
    print(f"Downloading {country} prices {start} -> {end} ...")
    return client.query_day_ahead_prices(country, start=start_ts, end=end_ts)


# The German bidding zone changed on 1 Oct 2018:
#   5 Jan 2015 - 30 Sep 2018 : DE-AT-LU (Germany + Austria + Luxembourg)
#   since 1 Oct 2018         : DE-LU    (Germany + Luxembourg)
FIRST_DAY = "2015-01-05"      # first day-ahead price on the ENTSO-E platform
ZONE_SWITCH = "2018-10-01"


def download_range(start, end):
    """Download any period from 5 Jan 2015 on, using the right bidding zone for each part.
    start / end are date texts 'YYYY-MM-DD' (end exclusive)."""
    start, end = max(start, FIRST_DAY), end
    parts = []
    pieces = [("DE_AT_LU", start, min(end, ZONE_SWITCH)), ("DE_LU", max(start, ZONE_SWITCH), end)]
    for zone, a, b in pieces:
        if a >= b:
            continue
        try:
            parts.append(download_prices(start=a, end=b, country=zone))
        except NoMatchingDataError:
            print(f"No {zone} prices for {a} -> {b}")
    if not parts:
        raise NoMatchingDataError("ENTSO-E returned no prices for this period")
    return pd.concat(parts)


def add_prices(start, end):
    """Download a period and MERGE it with the prices already saved (for the app).
    Returns the new clean series (Berlin time)."""
    new = download_range(start, end)
    new.index = new.index.tz_convert("UTC")
    # Keep what we already have (raw file, or the clean file on a fresh deployment)
    old_file = RAW_FILE if RAW_FILE.exists() else CLEAN_FILE if CLEAN_FILE.exists() else None
    if old_file is not None:
        old = pd.read_csv(old_file, index_col=0)["price_eur_mwh"]
        old.index = pd.to_datetime(old.index, utc=True)
        new = pd.concat([old, new])
    combined = new[~new.index.duplicated(keep="last")].sort_index()
    combined.to_csv(RAW_FILE, header=["price_eur_mwh"])
    clean = clean_prices(combined)
    clean.to_csv(CLEAN_FILE, index_label="timestamp_utc")
    print(f"Saved {len(clean):,} hours: {clean.index.min()} -> {clean.index.max()}")
    return load_prices()


def clean_prices(prices):
    """Make a clean, regular, hourly UTC series without gaps or duplicates."""
    s = prices.copy()                         # work on a copy, keep the original safe

    # 1. Convert all timestamps to UTC (UTC has no daylight-saving jumps)
    s.index = s.index.tz_convert("UTC")

    # 2. Remove duplicate timestamps
    n_dup = s.index.duplicated().sum()
    s = s[~s.index.duplicated(keep="first")]

    # 3. Sort by time
    s = s.sort_index()

    # 4. Force a regular hourly grid (15-min values are averaged; gaps become NaN)
    s = s.resample("h").mean()

    # 5. Fill SHORT gaps (up to 6 hours) with a straight line between neighbours.
    #    Longer gaps (e.g. years that were never downloaded) are left out.
    n_missing = s.isna().sum()
    s = s.interpolate(limit=6, limit_area="inside")
    n_dropped = s.isna().sum()
    s = s.dropna()

    print(f"Removed {n_dup} duplicates, filled {n_missing - n_dropped} missing hours"
          + (f", {n_dropped:,} hours not downloaded" if n_dropped else ""))
    s.name = "price_eur_mwh"
    return s


def load_prices():
    """Load the cleaned prices and convert to Berlin time (for grouping into days)."""
    df = pd.read_csv(CLEAN_FILE, index_col=0)
    s = df["price_eur_mwh"]
    s.index = pd.to_datetime(s.index, utc=True).tz_convert(config.LOCAL_TZ)
    return s


def prices_from_csv(file):
    """Read a user-uploaded price file (for the Streamlit app).
    Expected: a CSV with a time column (first column, or one named 'timestamp')
    and a price column in EUR/MWh (named 'price_eur_mwh' or 'price', or the second column).
    Times without a timezone are treated as Berlin local time.
    Returns a clean hourly series in Berlin time, like load_prices()."""
    df = pd.read_csv(file)
    time_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    price_cols = [c for c in df.columns if c.lower() in ("price_eur_mwh", "price")]
    price_col = price_cols[0] if price_cols else df.columns[1]

    try:
        times = pd.to_datetime(df[time_col])
    except (ValueError, TypeError):                 # mixed +01:00 / +02:00 offsets
        times = pd.to_datetime(df[time_col], utc=True)
    if times.dt.tz is None:
        # Local times: let pandas decide summer/winter time, drop impossible DST hours
        try:
            times = times.dt.tz_localize(config.LOCAL_TZ, ambiguous="infer",
                                         nonexistent="shift_forward")
        except Exception:
            times = times.dt.tz_localize(config.LOCAL_TZ, ambiguous="NaT",
                                         nonexistent="shift_forward")
    else:
        times = times.dt.tz_convert(config.LOCAL_TZ)
    s = pd.Series(pd.to_numeric(df[price_col], errors="coerce").values,
                  index=pd.DatetimeIndex(times))
    s = s[s.index.notna()].dropna()
    if s.empty:
        raise ValueError("No valid prices found in the uploaded file")
    s = clean_prices(s)
    s.index = s.index.tz_convert(config.LOCAL_TZ)
    return s


def full_download():
    """First run: download the whole period, clean it and save both files."""
    raw = download_prices()
    raw.to_csv(RAW_FILE, header=["price_eur_mwh"])
    clean = clean_prices(raw)
    clean.to_csv(CLEAN_FILE, index_label="timestamp_utc")
    print(f"Saved {len(clean):,} hours, last = {clean.index.max()}")
    return clean


def update_prices():
    """Later runs: download only the days missing since the last update, then re-clean."""
    # 1. Load what we already have (raw file, timestamps -> UTC)
    old = pd.read_csv(RAW_FILE, index_col=0)["price_eur_mwh"]
    old.index = pd.to_datetime(old.index, utc=True)

    # 2. Start from the day of the last timestamp we have (small overlap is fine)
    last_day = old.index.max().tz_convert(config.LOCAL_TZ).strftime("%Y-%m-%d")
    if last_day >= config.END:
        print("Already up to date.")
        return
    print(f"Last data: {last_day} -> downloading until {config.END}")

    # 3. Download only the missing part
    try:
        new = download_prices(start=last_day, end=config.END)
    except NoMatchingDataError:
        print("No new prices published yet (tomorrow's come out around 13:00).")
        return
    new.index = new.index.tz_convert("UTC")

    # 4. Glue old + new together, remove the overlap (keep the newest values)
    combined = pd.concat([old, new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_csv(RAW_FILE, header=["price_eur_mwh"])

    # 5. Clean again and save
    clean = clean_prices(combined)
    clean.to_csv(CLEAN_FILE, index_label="timestamp_utc")
    print(f"Updated: {len(clean):,} hours, last = {clean.index.max()}")


if __name__ == "__main__":
    # This block ONLY runs when you run this file directly.
    # It does NOT run when another file does "from data_loader import load_prices".

    # A. First run -> full download.  Later runs -> only the missing days.
    if RAW_FILE.exists():
        update_prices()
    else:
        full_download()

    # B. Sanity checks
    berlin = load_prices()
    print(berlin.describe())
    print("First hour:", berlin.index.min())
    print("Last hour :", berlin.index.max())

    # C. Day lengths in Berlin time (DST days should have 23 or 25 hours)
    hours_per_day = berlin.groupby(berlin.index.date).size()
    print(hours_per_day.value_counts())

    # D. Quick plot of all prices
    fig, ax = plt.subplots(figsize=(12, 4))
    berlin.plot(ax=ax, linewidth=0.5)
    ax.set_title("DE-LU Day-Ahead Prices")
    ax.set_ylabel("EUR/MWh")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "00_price_overview.png", dpi=150)
    plt.show()