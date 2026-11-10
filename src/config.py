# config.py
# -------------------------------------------------------------
# PURPOSE: Store ALL project settings in one place.
# Other files do "import config" and use e.g. config.P_MAX
# -------------------------------------------------------------
from pathlib import Path
from datetime import date, timedelta

# ---------- 1. Folder paths ----------
# Find the project root folder (BESS), no matter where Python is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"              # prices as downloaded
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"  # cleaned prices
DATA_WEATHER = PROJECT_ROOT / "data" / "weather"      # NASA POWER weather cache
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"        # PNG graphs
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"    # CSV results
IMAGES_DIR = PROJECT_ROOT / "images_app"              # pictures for the Streamlit app
API_KEY_FILE = PROJECT_ROOT / "entsoe_api_key.txt"    # my ENTSO-E key

# Create the folders if they don't exist yet (no error if they already exist)
for folder in [DATA_RAW, DATA_PROCESSED, DATA_WEATHER, FIG_DIR, RESULTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------- 2. Market data settings ----------
COUNTRY_CODE = "DE_LU"        # Germany-Luxembourg bidding zone
START = "2023-01-01"          # first day to download
END = (date.today() + timedelta(days=2)).isoformat()   # exclusive -> up to tomorrow (if published)
LOCAL_TZ = "Europe/Berlin"    # used to group hours into trading days
DT = 1.0                      # time step in hours (1.0 = hourly)

# ---------- 3. Battery settings (Datasheet and Assumptions) ----------
P_MAX = 1.0                   # max charge/discharge power (MW)
E_NOM = 4.0                   # nominal energy capacity (MWh)
ETA_RT = 0.88                 # round-trip efficiency (88 %)
SOC_MIN_FRAC = 0.1            # never below 10 % state of charge
SOC_MAX_FRAC = 0.9            # never above 90 %
SOC_INIT_FRAC = 0.5           # start (and end) each day at 50 %

# ---------- 4. Degradation settings (Datasheet and Assumptions) ----------
CAPEX_EUR_PER_KWH = 200       # battery replacement cost
WOEHLER_A = 3000              # cycles to End-of-Life at 100 % DoD
WOEHLER_B = 1.4               # Woehler exponent
SOH_EOL = 0.8                 # End-of-Life = 80 % State of Health
CAL_LIFE_YEARS = 15           # calendar life (aging even when idle)

# ---------- 5. Asset configuration ----------
# "grid"  = standalone battery, charges from the grid at any hour
# "solar" = solar park + battery (battery charges from the PV park)
# "wind"  = wind park + battery  (battery charges from the wind park)
ASSET_MODE = "grid"
ASSET_MODES = ["grid", "solar", "wind"]
ALLOW_GRID_CHARGING = False   # solar/wind only: may the battery ALSO charge from the grid?
GRID_CONNECTION_MW = 2.0      # solar/wind only: max export/import at the grid connection (MW)

# ---------- 6. Weather location (NASA POWER, free, no API key) ----------
LATITUDE = 51.16              # centre of Germany
LONGITUDE = 10.45

# ---------- 7. Solar park (PV) ----------
PV_PEAK_MW = 2.0              # installed PV size (MWp)
PV_PR = 0.80                  # performance ratio (inverter, heat, soiling losses)

# ---------- 8. Wind park ----------
WIND_RATED_MW = 2.0           # installed wind power (MW)
WIND_HUB_HEIGHT_M = 100       # turbine hub height (m)
WIND_CUT_IN = 3.0             # m/s - below this the turbine produces nothing
WIND_RATED_SPEED = 12.0       # m/s - full power from this speed
WIND_CUT_OUT = 25.0           # m/s - turbine shuts down above this speed
WIND_SHEAR_ALPHA = 0.143      # power-law exponent to scale 50 m wind to hub height


# ---------- 9. Helper: read the API key ----------
def read_api_key():
    """Read the ENTSO-E API key: environment variable ENTSOE_API_KEY (used on Render),
    otherwise the file entsoe_api_key.txt in the project root."""
    import os
    if os.environ.get("ENTSOE_API_KEY", "").strip():
        return os.environ["ENTSOE_API_KEY"].strip()
    if not API_KEY_FILE.exists():
        raise FileNotFoundError(f"API key file not found: {API_KEY_FILE}")
    key = API_KEY_FILE.read_text().strip()
    # If the file contains "ENTSOE_API_KEY=xxxx", keep only the part after "="
    if "=" in key:
        key = key.split("=", 1)[1].strip()
    if not key:
        raise ValueError("entsoe_api_key.txt is empty - paste your key into it")
    return key
