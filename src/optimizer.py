# optimizer.py
# -------------------------------------------------------------
# STEP 4: Daily MILP dispatch (PuLP + HiGHS solver)
#
# Three asset configurations (same model, different inputs):
#   grid  : standalone battery, gen = 0, charges from the grid
#   solar : PV park + battery, gen = PV output (MW)
#   wind  : wind park + battery, gen = wind output (MW)
#
# For solar/wind the park output G(t) is split every hour into:
#   G(t) = P_res_sell(t) + P_ch_res(t) + P_curt(t)
#          sold to grid    into battery   thrown away (e.g. negative prices)
# If grid charging is NOT allowed, the battery can only charge what the
# park produces in that hour ("solar bound" / "wind bound").
# -------------------------------------------------------------
import numpy as np
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import config
from degradation import marginal_deg_cost


def new_var(prob, name, **kw):
    """Create a variable in both old (2.x/3.x) and new (4.x) PuLP versions."""
    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, **kw)
    return pulp.LpVariable(name, **kw)


def solver_status(result):
    """Status text ('Optimal', 'Infeasible', ...) for old and new PuLP versions."""
    if hasattr(result, "status"):                    # PuLP 4: result object
        return result.status.name
    return pulp.LpStatus[result]                     # PuLP 2/3: a number


def park_alone_revenue(prices, gen, dt=config.DT, p_grid=None):
    """Revenue of the solar/wind park WITHOUT a battery:
    sell everything (up to the grid connection) when price >= 0, curtail at negative prices."""
    prices, gen = np.asarray(prices, dtype=float), np.asarray(gen, dtype=float)
    sold = gen if p_grid is None else np.minimum(gen, p_grid)
    sold = np.where(prices >= 0, sold, 0.0)
    return float((prices * sold * dt).sum())


def run_bess_optimization(prices, dt=config.DT, p_max=config.P_MAX, e_max=config.E_NOM,
                          eta_rt=config.ETA_RT, capex_per_kwh=config.CAPEX_EUR_PER_KWH,
                          soc_min_frac=config.SOC_MIN_FRAC, soc_max_frac=config.SOC_MAX_FRAC,
                          soc_init_frac=config.SOC_INIT_FRAC, c_deg=None, verbose=True,
                          gen=None, allow_grid_charging=True, p_grid=None):
    """Optimise one day.
    prices              : price per time step (EUR/MWh)
    c_deg               : None -> calculate it; 0 -> naive model
    gen                 : renewable output per time step (MW); None -> standalone battery
    allow_grid_charging : may the battery buy from the grid? (always True without gen)
    p_grid              : grid connection limit (MW) for export and import; None -> no limit
    Returns (df, trading_profit, penalty):
      trading_profit = extra market revenue thanks to the battery
                       (standalone: buy/sell profit; hybrid: plant revenue - park-alone revenue)
      penalty        = degradation penalty (EUR)
    More numbers are stored in df.attrs (status, total_revenue, park_alone_revenue, c_deg)."""
    # ---- 1. Setup ----
    prices = [float(x) for x in prices]
    T = len(prices)                                   # 24 normally, 23/25 on DST days
    hybrid = gen is not None                          # solar/wind park next to the battery
    gen = [0.0] * T if gen is None else [max(0.0, float(g)) for g in gen]
    if not hybrid:
        allow_grid_charging = True                    # a standalone battery must use the grid
    eta_ch = eta_dis = np.sqrt(eta_rt)                # split losses equally
    soc_min, soc_max = soc_min_frac * e_max, soc_max_frac * e_max
    soc_init = soc_init_frac * e_max
    if c_deg is None:
        c_deg = marginal_deg_cost(capex_per_kwh, dod_ref=soc_max_frac - soc_min_frac)
    grid_cap = p_grid if p_grid is not None else None

    prob = pulp.LpProblem("BESS_Arbitrage", pulp.LpMaximize)

    # ---- 2. Decision variables (one per time step) ----
    ch_grid_ub = p_max if allow_grid_charging else 0.0
    if grid_cap is not None:
        ch_grid_ub = min(ch_grid_ub, grid_cap)
    p_ch_grid = [new_var(prob, f"p_ch_grid_{t}", lowBound=0, upBound=ch_grid_ub) for t in range(T)]
    p_ch_res = [new_var(prob, f"p_ch_res_{t}", lowBound=0, upBound=min(p_max, gen[t])) for t in range(T)]
    p_dis = [new_var(prob, f"p_dis_{t}", lowBound=0, upBound=p_max) for t in range(T)]
    p_sell = [new_var(prob, f"p_res_sell_{t}", lowBound=0, upBound=gen[t]) for t in range(T)]
    p_curt = [new_var(prob, f"p_curt_{t}", lowBound=0, upBound=gen[t]) for t in range(T)]
    soc = [new_var(prob, f"soc_{t}", lowBound=soc_min, upBound=soc_max) for t in range(T)]
    u = [new_var(prob, f"u_{t}", cat="Binary") for t in range(T)]   # 1=may charge, 0=may discharge
    p_ch = [p_ch_grid[t] + p_ch_res[t] for t in range(T)]            # total charging power

    # ---- 3. Objective: market revenue - degradation penalty ----
    # Money in : battery discharge + park energy sold directly
    # Money out: energy bought from the grid to charge the battery
    revenue = pulp.lpSum(prices[t] * (p_dis[t] + p_sell[t]) * dt for t in range(T))
    charging_cost = pulp.lpSum(prices[t] * p_ch_grid[t] * dt for t in range(T))
    deg_penalty = pulp.lpSum(c_deg * (p_ch[t] + p_dis[t]) * dt for t in range(T))
    prob += revenue - charging_cost - deg_penalty, "Total_Profit"

    # ---- 4. Constraints ----
    for t in range(T):
        prob += p_ch[t] <= p_max * u[t]                          # charge only if u = 1
        prob += p_dis[t] <= p_max * (1 - u[t])                   # discharge only if u = 0
        prob += p_sell[t] + p_ch_res[t] + p_curt[t] == gen[t]    # every park MW goes somewhere
        if grid_cap is not None:
            prob += p_sell[t] + p_dis[t] <= grid_cap             # export limit of the connection
        soc_prev = soc_init if t == 0 else soc[t - 1]
        prob += soc[t] == soc_prev + p_ch[t] * eta_ch * dt - p_dis[t] / eta_dis * dt
    prob += soc[T - 1] >= soc_init                               # end the day where it started

    # ---- 5. Solve ----
    status = solver_status(prob.solve(pulp.HiGHS(msg=False)))
    if status != "Optimal":
        raise RuntimeError(f"Optimisation failed: {status}")

    # ---- 6. Collect results in a table ----
    val = lambda xs: [float(pulp.value(x)) for x in xs]
    df = pd.DataFrame({
        "Price": prices,
        "Gen": gen,
        "P_ch_grid": val(p_ch_grid),
        "P_ch_res": val(p_ch_res),
        "P_dis": val(p_dis),
        "P_res_sell": val(p_sell),
        "P_curt": val(p_curt),
        "SoC": val(soc),
    })
    df["P_ch"] = df["P_ch_grid"] + df["P_ch_res"]
    df["Net_Power"] = df["P_dis"] - df["P_ch"]                   # battery: + discharge / - charge
    df["Grid_Export"] = df["P_dis"] + df["P_res_sell"] - df["P_ch_grid"]   # whole plant at the grid
    power_cols = df.columns.drop("Price")
    df[power_cols] = df[power_cols].where(df[power_cols].abs() > 1e-9, 0.0)   # remove 1e-12 noise

    total_revenue = float((df["Price"] * df["Grid_Export"] * dt).sum())
    alone = park_alone_revenue(prices, gen, dt, grid_cap) if hybrid else 0.0
    trading_profit = total_revenue - alone
    penalty = float(pulp.value(deg_penalty))
    df.attrs.update(status=status, c_deg=c_deg, total_revenue=total_revenue,
                    park_alone_revenue=alone, trading_profit=trading_profit, penalty=penalty)

    if verbose:
        print(f"Status: {status} | c_deg = {c_deg:.2f} EUR/MWh")
        if hybrid:
            print(f"Plant revenue {total_revenue:,.2f} | Park alone {alone:,.2f} EUR")
        print(f"Battery profit {trading_profit:,.2f} | Penalty {penalty:,.2f} | "
              f"Net {trading_profit - penalty:,.2f} EUR")
    return df, trading_profit, penalty


def plot_day(df, title, filename):
    """Price on top, battery power + SoC (+ renewable output) below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.step(df.index, df["Price"], where="mid")
    ax1.set_ylabel("Price (EUR/MWh)")
    ax1.set_title(title)
    colors = np.where(df["Net_Power"] > 0, "green", "red")
    ax2.bar(df.index, df["Net_Power"], color=colors, label="Net power (+ discharge / - charge)")
    ax2.plot(df.index, df["SoC"], "k-o", label="SoC (MWh)")
    if df["Gen"].sum() > 0:
        ax2.fill_between(df.index, df["Gen"], color="gold", alpha=0.3, step="mid",
                         label="Renewable output (MW)")
    ax2.set_xlabel("Hour")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / filename, dpi=150)


if __name__ == "__main__":
    # ---- Test 1: synthetic day with a BIG spread -> battery should trade ----
    np.random.seed(42)
    big = (80 + 60 * np.sin(np.linspace(0, 4 * np.pi, 24)) + np.random.normal(0, 10, 24)).tolist()
    df1, _, _ = run_bess_optimization(big)
    plot_day(df1, "Synthetic day - big spread", "01_dispatch_big_spread.png")

    # ---- Test 2: SMALL spread (below 57.3 threshold) -> battery should stay idle ----
    small = (80 + 10 * np.sin(np.linspace(0, 2 * np.pi, 24))).tolist()
    df2, _, _ = run_bess_optimization(small)
    print("Throughput on small-spread day:", round((df2.P_ch + df2.P_dis).sum(), 6), "(expect ~0)")

    # ---- Test 3: same small day, NAIVE (c_deg=0) -> it WILL trade ----
    run_bess_optimization(small, c_deg=0.0)

    # ---- Test 4: one real day from your ENTSO-E data ----
    from data_loader import load_prices
    prices = load_prices()
    day = prices.loc["2024-06-15"]
    df4, _, _ = run_bess_optimization(day.values)
    plot_day(df4, "Real day 2024-06-15 (DE-LU)", "02_dispatch_real_day.png")

    # ---- Test 5: same day, SOLAR + battery, battery may only charge from PV ----
    from weather import generation_profile
    pv = generation_profile(day.index, mode="solar")["gen_mw"].values
    df5, _, _ = run_bess_optimization(day.values, gen=pv, allow_grid_charging=False,
                                      p_grid=config.GRID_CONNECTION_MW)
    night = df5["Gen"] == 0
    print("Grid charging used:", round(df5["P_ch_grid"].sum(), 6), "(expect 0)")
    print("Charging in hours without sun:", round(df5.loc[night, "P_ch"].sum(), 6), "(expect 0)")
    plot_day(df5, "Real day 2024-06-15 - Solar + battery", "02b_dispatch_solar_day.png")
    plt.show()
