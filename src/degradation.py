# degradation.py
# -------------------------------------------------------------
# STEP 2: Woehler curve (cycle life vs depth of discharge)
# STEP 3: Marginal degradation cost (EUR/MWh) + trading threshold
# Helper for STEP 5: SoH loss from a SoC profile (rainflow counting)
# -------------------------------------------------------------
import numpy as np
import rainflow
import config

def cycle_life(dod, a=config.WOEHLER_A, b=config.WOEHLER_B):
    """Number of cycles until End-of-Life at a given depth of discharge.
    dod must be a FRACTION (0.5), NOT a percent (50)!"""
    return a * dod ** (-b)


def fit_woehler_b(n1, dod1, n2, dod2):
    """Find b from two datasheet points: n1 cycles at dod1, n2 cycles at dod2."""
    return np.log(n2 / n1) / np.log(dod1 / dod2)

def marginal_deg_cost(capex_eur_per_kwh=config.CAPEX_EUR_PER_KWH,
                      dod_ref=config.SOC_MAX_FRAC - config.SOC_MIN_FRAC,
                      a=config.WOEHLER_A, b=config.WOEHLER_B, salvage_frac=0.0):
    """Degradation cost in EUR per MWh, applied to EACH direction (charge and discharge)."""
    # 1. EUR/kWh -> EUR/MWh (x1000), minus any resale value
    capex_eur_per_mwh = capex_eur_per_kwh * 1000 * (1 - salvage_frac)
    # 2. Lifetime throughput per MWh of capacity = 2 * DoD * N_cycles
    return capex_eur_per_mwh / (2 * dod_ref * cycle_life(dod_ref, a, b))

def trading_threshold(c_deg, eta_rt=config.ETA_RT):
    """Minimum value of (eta_rt * sell_price - buy_price) for a trade to pay off."""
    return c_deg * (1 + eta_rt)

def soh_loss(soc_path_mwh, e_nom, days=1.0, soh_eol=config.SOH_EOL,
             a=config.WOEHLER_A, b=config.WOEHLER_B, cal_life_years=config.CAL_LIFE_YEARS):
    """SoH lost in one horizon = cycle aging (rainflow) + calendar aging."""
    # 1. SoC in MWh -> fraction of NOMINAL capacity (0..1)
    dod_series = np.asarray(soc_path_mwh) / e_nom

    # 2. Rainflow counting: find every (half) cycle and its depth
    cyc = 0.0
    for rng, mean, count, i0, i1 in rainflow.extract_cycles(dod_series):
        if rng > 1e-6:                              # ignore tiny numerical wiggles
            cyc += count / cycle_life(rng, a, b)    # count = 1 (full) or 0.5 (half)

    # 3. Calendar aging: a straight line over the calendar life
    cal = days / (cal_life_years * 365)

    # 4. From 100 % to EoL (80 %) there is 0.2 of SoH to "use up"
    return (1 - soh_eol) * (cyc + cal)

if __name__ == "__main__":
    # ---- Check against the numbers in the guide ----
    print("Cycles at 100% DoD:", round(cycle_life(1.0)))       # expect 3000
    print("Cycles at 50% DoD :", round(cycle_life(0.5)))       # expect ~7900
    print("Cycles at 80% DoD :", round(cycle_life(0.8)))       # expect ~4100
    print("Fitted b:", round(fit_woehler_b(3000, 1.0, 7918, 0.5), 2))  # expect 1.4

    c = marginal_deg_cost()
    print(f"Degradation cost: {c:.1f} EUR/MWh per direction")         # expect ~30.5
    print(f"Trading threshold: {trading_threshold(c):.1f} EUR/MWh")   # expect ~57.3

    # One full cycle 10% -> 90% -> 10% of a 4 MWh battery
    print("SoH loss for 1 deep cycle:", soh_loss([0.4, 3.6, 0.4], e_nom=4.0, days=0))