# test_api.py
# -------------------------------------------------------------
# PURPOSE: Check that my ENTSO-E key works (download 1 week only)
# -------------------------------------------------------------
import pandas as pd
from entsoe import EntsoePandasClient
import config

# 1. Read the key from entsoe_api_key.txt (function in config.py)
api_key = config.read_api_key()
print("Key loaded, length:", len(api_key))   # never print the key itself, only the length/validity

# 2. Create a "client" = an object that talks to the ENTSO-E server for us
client = EntsoePandasClient(api_key=api_key)

# 3. Start and end times. ENTSO-E requires a timezone.
start = pd.Timestamp("2024-01-01", tz="Europe/Berlin")
end = pd.Timestamp("2024-01-08", tz="Europe/Berlin")

# 4. Download day-ahead prices for Germany-Luxembourg
prices = client.query_day_ahead_prices("DE_LU", start=start, end=end)

# 5. Look at the result
print(prices.head(10))                      # first 10 rows
print("Number of values:", len(prices))     # ~168 = 7 days x 24 hours
print("Data type:", type(prices))