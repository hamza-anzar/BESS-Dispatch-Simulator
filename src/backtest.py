# backtest.py
# -------------------------------------------------------------
# STEP 5: Rolling daily simulation with State-of-Health tracking
# Works for all three asset configurations: grid, solar, wind
# -------------------------------------------------------------
import numpy as np
import pandas as pd
import config
from optimizer import run_bess_optimization
from degradation import soh_loss, marginal_deg_cost
from data_loader import load_prices


def backtest(prices, naive=False, n_laps=1, capex_per_kwh=config.CAPEX_EUR_PER_KWH,
             hourly_file=None, mode="grid", gen=None,
             allow_grid_charging=config.ALLOW_GRID_CHARGING, p_grid=config.GRID_CONNECTION_MW,
             soh_start=1.0, e_nom=config.E_NOM, woehler_a=config.WOEHLER_A,
             woehler_b=config.WOEHLER_B, soh_eol=config.SOH_EOL,
             cal_life_years=config.CAL_LIFE_YEARS, progress=None, **opt_kw):
    """Simulate day after day and track the battery's State of Health.
    naive=True    -> ignore degradation cost (c_deg = 0)
    n_laps        -> replay the price data several times to simulate a full battery life
    hourly_file   -> if given, save every hour's decisions to this CSV
    mode          -> 'grid', 'solar' or 'wind'
    gen           -> renewable output in MW for every price hour (solar/wind);
                     None -> taken from NASA POWER weather automatically
    soh_start     -> State of Health at the first day (1.0 = new battery)
    progress      -> optional function(fraction 0..1), e.g. a Streamlit progress bar
    opt_kw        -> extra settings for run_bess_optimization (p_max, eta_rt, soc_min_frac ...)"""
    # A. Renewable generation for every hour (zeros / None for a standalone battery)
    if mode != "grid" and gen is None:
        from weather import generation_profile
        gen = generation_profile(prices.index, mode=mode)["gen_mw"]
    if mode == "grid":
        gen, p_grid = None, None

    # B. Degradation cost per MWh (0 for the naive model)
    dod_ref = opt_kw.get("soc_max_frac", config.SOC_MAX_FRAC) - opt_kw.get("soc_min_frac", config.SOC_MIN_FRAC)
    c_deg = 0.0 if naive else marginal_deg_cost(capex_per_kwh, dod_ref, woehler_a, woehler_b)
    soc_init_frac = opt_kw.get("soc_init_frac", config.SOC_INIT_FRAC)
    dt = opt_kw.get("dt", config.DT)

    soh = soh_start                                    # C. battery health at the start
    rows = []                                          #    one row per DAY
    hourly_parts = []                                  #    one table per day, with all its HOURS
    stop = False                                       #    becomes True at End-of-Life
    sim_day = 0
    days = list(prices.groupby(prices.index.date))     # D. split into Berlin trading days
    n_total = len(days) * n_laps

    for lap in range(n_laps):
        for day, p in days:
            # 1. Capacity shrinks as the battery ages
            e_avail = e_nom * soh
            g = None if gen is None else gen.loc[p.index].values

            # 2. Solve today's optimisation
            df, profit, pen = run_bess_optimization(
                p.values, e_max=e_avail, c_deg=c_deg, verbose=False, gen=g,
                allow_grid_charging=allow_grid_charging, p_grid=p_grid, **opt_kw)

            # 3. SoC path (including the starting point) -> SoH loss
            soc_path = np.r_[soc_init_frac * e_avail, df["SoC"].values]
            deg_kw = dict(soh_eol=soh_eol, a=woehler_a, b=woehler_b, cal_life_years=cal_life_years)
            d_soh = soh_loss(soc_path, e_nom, days=len(p) * dt / 24, **deg_kw)
            d_soh_cyc = soh_loss(soc_path, e_nom, days=0, **deg_kw)   # part caused by cycling
            soh -= d_soh
            sim_day += 1

            # 4. Save today's results as one row
            # Health cost = share of the battery's useful life used up x replacement cost.
            #   cycling part  -> caused by today's trading decisions (counts in net profit)
            #   calendar part -> happens anyway, even when the battery is idle
            eur_per_soh = capex_per_kwh * 1000 * e_nom / (1 - soh_eol)
            health_cost = d_soh_cyc * eur_per_soh
            calendar_cost = (d_soh - d_soh_cyc) * eur_per_soh
            ch = df["P_ch"] > 1e-6
            dis = df["P_dis"] > 1e-6
            rows.append({
                "sim_day": sim_day, "lap": lap, "date": day, "mode": mode,
                "trading_profit": profit, "deg_penalty": pen,
                "health_cost": health_cost,
                "calendar_cost": calendar_cost,
                "net_profit": profit - health_cost,
                "total_revenue": df.attrs["total_revenue"],
                "park_alone_revenue": df.attrs["park_alone_revenue"],
                "throughput_mwh": (df["P_ch"] + df["P_dis"]).sum() * dt,
                "charged_mwh": df["P_ch"].sum() * dt,
                "discharged_mwh": df["P_dis"].sum() * dt,
                "gen_mwh": df["Gen"].sum() * dt,
                "curtailed_mwh": df["P_curt"].sum() * dt,
                "grid_charge_mwh": df["P_ch_grid"].sum() * dt,
                "avg_charge_price": (df.loc[ch, "Price"] * df.loc[ch, "P_ch"]).sum() / df.loc[ch, "P_ch"].sum() if ch.any() else np.nan,
                "avg_discharge_price": (df.loc[dis, "Price"] * df.loc[dis, "P_dis"]).sum() / df.loc[dis, "P_dis"].sum() if dis.any() else np.nan,
                "price_min": p.min(), "price_max": p.max(),
                "spread": opt_kw.get("eta_rt", config.ETA_RT) * p.max() - p.min(),
                "soh_loss": d_soh,
                "soh": soh,
            })

            # 4b. Also keep today's hourly decisions
            hourly = df.copy()
            hourly.index = p.index                     # real timestamps instead of 0, 1, 2 ... 23
            hourly["lap"] = lap
            hourly["E_avail"] = e_avail                # capacity used for this day (MWh)
            hourly["SoH"] = soh                        # SoH at the END of this day
            hourly_parts.append(hourly)

            if progress is not None:
                progress(sim_day / n_total)

            # 5. Stop at End-of-Life
            if soh <= soh_eol:
                stop = True
                break                                  # leave the inner (day) loop

            # 6. Progress message every 100 days
            if progress is None and sim_day % 100 == 0:
                print(f"  day {sim_day}: SoH = {soh:.4f}")

        if stop:
            break                                      # leave the outer (lap) loop too

    hourly_all = pd.concat(hourly_parts) if hourly_parts else pd.DataFrame()

    # 7. Save the hourly data (only if a file name was given)
    if hourly_file is not None:
        hourly_all.to_csv(hourly_file, index_label="timestamp")

    result = pd.DataFrame(rows)
    result.attrs["hourly"] = hourly_all                # the app reads the hours from here
    return result


if __name__ == "__main__":
    from weather import generation_profile
    prices = load_prices()

    for mode in config.ASSET_MODES:
        gen = None if mode == "grid" else generation_profile(prices.index, mode=mode)["gen_mw"]
        for naive in (False, True):
            name = "naive" if naive else "aware"
            print(f"Running {mode} / {name} model ...")
            res = backtest(prices, naive=naive, n_laps=1, mode=mode, gen=gen,   # later try n_laps=8
                           hourly_file=config.RESULTS_DIR / f"hourly_{mode}_{name}.csv")
            res.to_csv(config.RESULTS_DIR / f"backtest_{mode}_{name}.csv", index=False)
            print(f"{mode}/{name}: battery profit = {res.trading_profit.sum():,.0f} EUR, "
                  f"final SoH = {res.soh.iloc[-1]:.4f}, days = {len(res)}")
