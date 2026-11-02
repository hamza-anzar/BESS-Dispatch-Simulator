# dispatchplots.py
# -------------------------------------------------------------
# Hourly timing of trading: when does the battery charge / discharge?
# Needs hourly_<mode>_aware.csv and hourly_<mode>_naive.csv from backtest.py
# -------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config


def load_hourly(mode, name):
    """Load hourly backtest results and convert timestamps to Berlin time."""
    df = pd.read_csv(config.RESULTS_DIR / f"hourly_{mode}_{name}.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(config.LOCAL_TZ)
    return df


# ============================================================
# CHART A: One day - price vs charging/discharging (naive vs smart)
# ============================================================
def plot_day_compare(aware, naive, date, filename, mode="grid"):
    a = aware.loc[date]                      # all hours of that day (smart)
    n = naive.loc[date]                      # all hours of that day (naive)
    x = np.arange(len(a))                    # 0, 1, 2 ... 23 for plotting

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # --- Panel 1: price, with smart battery's charge/discharge hours shaded ---
    ax = axes[0]
    ax.step(x, a["Price"], where="mid", color="black", label="Day-ahead price")
    for i in range(len(a)):
        if a["P_ch"].iloc[i] > 0.01:
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.15)      # charging hour
        elif a["P_dis"].iloc[i] > 0.01:
            ax.axvspan(i - 0.5, i + 0.5, color="green", alpha=0.15)    # discharging hour
    ax.set_ylabel("Price [EUR/MWh]")
    ax.set_title(f"{date} ({mode}) - shaded: smart battery charging (red) / discharging (green)")
    ax.legend(loc="upper left")

    # --- Panel 2: net power, smart and naive side by side (+ renewable output) ---
    ax = axes[1]
    if a["Gen"].sum() > 0:
        ax.fill_between(x, a["Gen"], step="mid", color="gold", alpha=0.3, label=f"{mode} output")
    ax.bar(x - 0.2, a["Net_Power"], width=0.4, label="Degradation-aware")
    ax.bar(x + 0.2, n["Net_Power"], width=0.4, label="Naive", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Power [MW]\n(+ discharge / - charge)")
    ax.legend(loc="upper left")

    # --- Panel 3: state of charge ---
    ax = axes[2]
    ax.plot(x, a["SoC"], "-o", label="Degradation-aware")
    ax.plot(x, n["SoC"], "-o", label="Naive", alpha=0.6)
    ax.set_ylabel("SoC [MWh]")
    ax.set_xlabel("Hour of day")
    ax.set_xticks(x)
    ax.set_xticklabels(a.index.strftime("%H"))
    ax.legend(loc="upper left")

    # --- Daily battery profit in the window title (plant revenue - park alone) ---
    def battery_profit(d):
        export = (d["Price"] * d["Grid_Export"] * config.DT).sum()
        alone = (d["Price"].clip(lower=0) * np.minimum(d["Gen"], config.GRID_CONNECTION_MW)
                 * (d["Price"] >= 0) * config.DT).sum() if mode != "grid" else 0.0
        return export - alone
    fig.suptitle(f"Battery profit: smart {battery_profit(a):,.0f} EUR | "
                 f"naive {battery_profit(n):,.0f} EUR")

    fig.tight_layout()
    fig.savefig(config.FIG_DIR / filename, dpi=150)


# ============================================================
# CHART B: Average day - when does the battery usually trade?
# ============================================================
def plot_avg_profile(df, name, filename):
    g = df.groupby(df.index.hour)            # group all rows by hour of day (0-23)
    hours = g["Price"].mean().index
    avg_price = g["Price"].mean()
    avg_ch = g["P_ch"].mean()
    avg_dis = g["P_dis"].mean()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    # Bars = battery (left y-axis)
    ax1.bar(hours, -avg_ch, color="red", alpha=0.6, label="Avg charging")
    ax1.bar(hours, avg_dis, color="green", alpha=0.6, label="Avg discharging")
    if df["Gen"].sum() > 0:
        ax1.plot(hours, g["Gen"].mean(), color="goldenrod", ls="--", label="Avg renewable output")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Avg power [MW]\n(+ discharge / - charge)")
    ax1.set_xticks(range(24))

    # Line = price (second y-axis on the right)
    ax2 = ax1.twinx()
    ax2.plot(hours, avg_price, "k-o", label="Avg price")
    ax2.set_ylabel("Avg price [EUR/MWh]")

    # Combine both legends into one box
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title(f"Average daily trading profile - {name}")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / filename, dpi=150)


# ============================================================
# CHART C: Heatmap - average net power by month and hour
# ============================================================
def plot_heatmap(df, name, filename):
    d = df.assign(month=df.index.month, hour=df.index.hour)
    table = d.pivot_table(index="month", columns="hour", values="Net_Power", aggfunc="mean")

    vmax = table.abs().max().max() or 1.0    # make the colour scale symmetric around 0
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(table, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_yticklabels([months[m - 1] for m in table.index])
    ax.set_xlabel("Hour of day")
    ax.set_title(f"When does the battery trade? (red = charge, green = discharge) - {name}")
    fig.colorbar(im, label="Avg net power [MW]")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / filename, dpi=150)


if __name__ == "__main__":
    for mode in config.ASSET_MODES:
        aware = load_hourly(mode, "aware")
        naive = load_hourly(mode, "naive")

        # --- Pick interesting days automatically ---
        daily = pd.read_csv(config.RESULTS_DIR / f"backtest_{mode}_aware.csv")
        best_day = daily.loc[daily["trading_profit"].idxmax(), "date"]      # most profitable
        idle = daily.loc[daily["throughput_mwh"] < 0.01, "date"]            # smart stayed idle
        neg_day = str(aware["Price"].idxmin().date())                       # lowest price hour

        plot_day_compare(aware, naive, best_day, f"07_day_best_{mode}.png", mode)
        if len(idle) > 0:
            plot_day_compare(aware, naive, idle.iloc[0], f"08_day_idle_{mode}.png", mode)
        plot_day_compare(aware, naive, neg_day, f"09_day_negative_price_{mode}.png", mode)

        plot_avg_profile(aware, f"Degradation-aware ({mode})", f"10_avg_profile_{mode}_aware.png")
        plot_avg_profile(naive, f"Naive ({mode})", f"11_avg_profile_{mode}_naive.png")

        plot_heatmap(aware, f"Degradation-aware ({mode})", f"12_heatmap_{mode}_aware.png")
        plot_heatmap(naive, f"Naive ({mode})", f"13_heatmap_{mode}_naive.png")
        plt.close("all")                       # free memory before the next mode
        print(f"{mode}: charts saved")
