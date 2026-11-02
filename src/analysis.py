# analysis.py
# -------------------------------------------------------------
# STEP 6: Portfolio plots + KPI summary (for grid, solar and wind)
# Needs backtest_<mode>_<aware|naive>.csv from backtest.py
# -------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import config
from degradation import marginal_deg_cost, trading_threshold
from backtest import backtest
from data_loader import load_prices


def load_results(mode, name):
    """Daily backtest results for one asset mode and one model ('aware' or 'naive')."""
    return pd.read_csv(config.RESULTS_DIR / f"backtest_{mode}_{name}.csv")


# ---------- PLOT 1: Threshold trading sensitivity ----------
def plot_threshold(aware, naive, mode):
    thr = trading_threshold(marginal_deg_cost())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(naive["spread"], naive["throughput_mwh"], s=8, alpha=0.4, label="Naive")
    ax.scatter(aware["spread"], aware["throughput_mwh"], s=8, alpha=0.6, label="Degradation-aware")
    ax.axvline(thr, color="red", ls="--", label=f"Threshold {thr:.1f} EUR/MWh")
    ax.set_xlabel("Daily spread (eta_rt * max price - min price) [EUR/MWh]")
    ax.set_ylabel("Daily throughput [MWh]")
    ax.set_title(f"Threshold sensitivity - {mode}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"03_threshold_sensitivity_{mode}.png", dpi=150)


# ---------- PLOT 2: How CAPEX shifts the threshold (grid mode, first 90 days) ----------
def plot_capex_sensitivity():
    prices = load_prices()
    first_90 = prices[prices.index < prices.index[0] + pd.Timedelta(days=90)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for capex in (100, 200, 300):
        r = backtest(first_90, capex_per_kwh=capex)
        ax.scatter(r["spread"], r["throughput_mwh"], s=10, label=f"CAPEX {capex} EUR/kWh")
    ax.set_xlabel("Daily spread [EUR/MWh]")
    ax.set_ylabel("Daily throughput [MWh]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "04_capex_sensitivity.png", dpi=150)


# ---------- PLOT 3: Cumulative profit (battery profit and net profit) ----------
def plot_cumulative(aware, naive, mode):
    fig, ax = plt.subplots(figsize=(10, 4))
    for r, name in ((naive, "Naive"), (aware, "Degradation-aware")):
        ax.plot(r["sim_day"], r["trading_profit"].cumsum(), label=f"{name} - battery profit")
        ax.plot(r["sim_day"], r["net_profit"].cumsum(), ls="--", label=f"{name} - after health cost")
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("Cumulative EUR")
    ax.set_title(f"Cumulative profit - {mode}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"05_cumulative_profit_{mode}.png", dpi=150)


# ---------- PLOT 4: SoH trajectory ----------
def plot_soh(aware, naive, mode):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(naive["sim_day"], naive["soh"] * 100, label="Naive")
    ax.plot(aware["sim_day"], aware["soh"] * 100, label="Degradation-aware")
    ax.axhline(config.SOH_EOL * 100, color="red", ls="--", label="End-of-Life")
    ax.set_xlabel("Simulated day")
    ax.set_ylabel("SoH [%]")
    ax.set_title(f"State of Health - {mode}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"06_soh_trajectory_{mode}.png", dpi=150)


# ---------- KPI table ----------
def kpis(r, mode, name):
    """Key numbers for one mode + model."""
    soh_lost_pct = (1 - r["soh"].iloc[-1]) * 100
    return {
        "mode": mode,
        "model": name,
        "battery_profit_eur": r["trading_profit"].sum(),
        "health_cost_eur": r["health_cost"].sum(),
        "net_profit_eur": r["net_profit"].sum(),
        "throughput_mwh": r["throughput_mwh"].sum(),
        "soh_lost_pct": soh_lost_pct,
        "profit_per_pct_soh": r["trading_profit"].sum() / soh_lost_pct,
        "days_simulated": len(r),
    }


if __name__ == "__main__":
    rows = []
    for mode in config.ASSET_MODES:
        aware = load_results(mode, "aware")
        naive = load_results(mode, "naive")
        plot_threshold(aware, naive, mode)
        plot_cumulative(aware, naive, mode)
        plot_soh(aware, naive, mode)
        rows += [kpis(naive, mode, "Naive"), kpis(aware, mode, "Degradation-aware")]

    plot_capex_sensitivity()

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "kpi_summary.csv", index=False)
    print(table.round(2).to_string(index=False))
    plt.show()      # show all charts at the end
