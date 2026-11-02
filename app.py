# app.py
# -------------------------------------------------------------
# BESS Dispatch Optimizer - Streamlit app (degradation-aware)
# Run from the project root:   streamlit run app.py
#
# Tabs:
#   1. Inputs      - asset type, battery, degradation, price data, weather
#   2. Results     - one day (24 h table + charts) or a date range (+ downloads)
#   3. Methodology - notes, formulas and assumptions, filled with the current inputs
# All calculations come from the files in src/ (optimizer, backtest, weather ...).
# -------------------------------------------------------------
import base64
import hashlib
import io
import sys
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))           # so we can import config, backtest ...

import config                                    # noqa: E402
from data_loader import load_prices, prices_from_csv, add_prices, CLEAN_FILE, FIRST_DAY   # noqa: E402
from weather import get_weather, weather_from_csv, generation_profile, generation_from_upload  # noqa: E402
from degradation import cycle_life, marginal_deg_cost, trading_threshold      # noqa: E402
from backtest import backtest                    # noqa: E402

st.set_page_config(page_title="BESS Dispatch Optimizer", page_icon="🔋", layout="wide")

# ---------- Look & feel ----------
C = dict(charge="#dc2626", charge_res="#f59e0b", discharge="#16a34a", idle="#9ca3af",
         gen="#facc15", soc="#2563eb", aware="#2563eb", naive="#f97316", price="#374151")
MODE_LABEL = {"grid": "🔌 Grid + Battery", "solar": "☀️ Solar + Battery", "wind": "🌬️ Wind + Battery"}
MODE_IMAGE = {"grid": "gridbattery.png", "solar": "solarbattery.png", "wind": "windbattery.png"}
MODE_TEXT = {
    "grid": "Standalone battery. Buys from and sells to the grid at any hour.",
    "solar": "Battery next to a solar park. Charges from the PV output "
             "(optionally also from the grid).",
    "wind": "Battery next to a wind park. Charges from the wind output "
            "(optionally also from the grid).",
}
MODEL_LABEL = {"aware": "🧠 Degradation-aware", "naive": "⚡ Naive (ignores wear)"}


def eur(x, dec=0):
    return "–" if x is None or pd.isna(x) else f"{round(x, dec) + 0.0:,.{dec}f} €"   # +0.0 avoids "-0 €"


def num(x, unit="", dec=2):
    return "–" if x is None or pd.isna(x) else f"{x:,.{dec}f}{(' ' + unit) if unit else ''}"


# ============================================================
# Cached data loading (runs once, then re-used until inputs change)
# ============================================================
@st.cache_data(show_spinner="Loading ENTSO-E prices ...")
def cached_base_prices(file_mtime):
    """file_mtime makes the cache refresh after the price file is updated."""
    return load_prices()


@st.cache_data(show_spinner="Reading uploaded prices ...")
def cached_upload_prices(data: bytes):
    return prices_from_csv(io.BytesIO(data))


@st.cache_data(show_spinner="Downloading NASA POWER weather (first time per location ~1 min) ...")
def cached_nasa(lat, lon, start):
    return get_weather(lat, lon, start=start)


@st.cache_data(show_spinner="Reading uploaded weather ...")
def cached_weather_upload(data: bytes):
    return weather_from_csv(io.BytesIO(data))


@st.cache_data(show_spinner="Calculating renewable output ...")
def cached_generation(prices, mode, weather, is_upload, kw_items):
    index = prices.index                       # (a Series can be hashed, a bare index cannot)
    kw = dict(kw_items)
    if is_upload:
        g = generation_from_upload(weather, index, mode, **kw)
    else:
        g = generation_profile(index, mode=mode, weather=weather, **kw)
    return g[["gen_mw", "filled"]]


@st.cache_data(show_spinner=False)
def cached_sim(prices, gen, naive, mode, params_items):
    """Run the backtest (also used for a single day). Returns (daily, hourly)."""
    res = backtest(prices, naive=naive, mode=mode, gen=gen, **dict(params_items))
    hourly = res.attrs.pop("hourly")
    return res, hourly


def run_sim(prices, gen, naive, mode, params, progress=None):
    """Like cached_sim, but with a progress bar (results stored in session_state)."""
    res = backtest(prices, naive=naive, mode=mode, gen=gen, progress=progress, **params)
    hourly = res.attrs.pop("hourly")
    return res, hourly


# ============================================================
# Helpers for tables and KPIs
# ============================================================
def hour_labels(idx):
    """'00:00' ... '23:00'; the repeated hour on the October clock change gets a suffix."""
    labels, seen = [], set()
    for t in idx:
        s = t.strftime("%H:%M")
        if s in seen:
            s = f"{s} ({t.strftime('%Z')})"
        seen.add(s)
        labels.append(s)
    return labels


def actions(h):
    return np.where(h["P_ch"] > 1e-6, "Charge", np.where(h["P_dis"] > 1e-6, "Discharge", "Idle"))


def kpis(res, h, p):
    """Key numbers for one model over a day or a period."""
    ch, dis = h["P_ch"] > 1e-6, h["P_dis"] > 1e-6
    dod = p["soc_max_frac"] - p["soc_min_frac"]
    w_avg = lambda mask, col: ((h.loc[mask, "Price"] * h.loc[mask, col]).sum() / h.loc[mask, col].sum()
                               if mask.any() else np.nan)
    return {
        "Battery profit": res["trading_profit"].sum(),
        "Health cost (cycling)": res["health_cost"].sum(),
        "Net profit": res["net_profit"].sum(),
        "Calendar aging cost": res["calendar_cost"].sum(),
        "Charged (MWh)": res["charged_mwh"].sum(),
        "Discharged (MWh)": res["discharged_mwh"].sum(),
        "Equivalent full cycles": res["discharged_mwh"].sum() / (p["e_nom"] * dod),
        "Lowest charging price": h.loc[ch, "Price"].min() if ch.any() else np.nan,
        "Highest discharging price": h.loc[dis, "Price"].max() if dis.any() else np.nan,
        "Avg charging price": w_avg(ch, "P_ch"),
        "Avg discharging price": w_avg(dis, "P_dis"),
        "SoH at end (%)": res["soh"].iloc[-1] * 100,
        "SoH lost (%-points)": res["soh_loss"].sum() * 100,
        "Plant revenue": res["total_revenue"].sum(),
        "Park alone revenue": res["park_alone_revenue"].sum(),
        "Renewable output (MWh)": res["gen_mwh"].sum(),
        "Curtailed (MWh)": res["curtailed_mwh"].sum(),
        "Grid charging (MWh)": res["grid_charge_mwh"].sum(),
    }


def show_kpis(k, hybrid, dec):
    c = st.columns(4)
    c[0].metric("Battery profit", eur(k["Battery profit"], dec),
                help="Extra market revenue thanks to the battery "
                     "(hybrid: plant revenue minus what the park earns alone).")
    c[1].metric("Health cost (cycling)", eur(k["Health cost (cycling)"], dec),
                help="Battery life used up by cycling, valued at the replacement cost.")
    c[2].metric("Net profit", eur(k["Net profit"], dec), help="Battery profit − health cost.")
    c[3].metric("SoH at end", num(k["SoH at end (%)"], "%", 3),
                delta=f"-{k['SoH lost (%-points)']:.4f} %-pts", delta_color="inverse")
    c = st.columns(4)
    c[0].metric("Charged (MWh)", num(k["Charged (MWh)"]))
    c[1].metric("Discharged (MWh)", num(k["Discharged (MWh)"]))
    c[2].metric("Lowest charging price (€/MWh)", num(k["Lowest charging price"]),
                help="Lowest market price in an hour where the battery charged.")
    c[3].metric("Highest discharging price (€/MWh)", num(k["Highest discharging price"]),
                help="Highest market price in an hour where the battery discharged.")
    c = st.columns(4)
    c[0].metric("Equivalent full cycles", num(k["Equivalent full cycles"]),
                help="Discharged energy ÷ usable capacity.")
    c[1].metric("Avg charging price (€/MWh)", num(k["Avg charging price"]),
                help="Energy-weighted average price paid while charging.")
    c[2].metric("Avg discharging price (€/MWh)", num(k["Avg discharging price"]),
                help="Energy-weighted average price received while discharging.")
    c[3].metric("Calendar aging cost", eur(k["Calendar aging cost"], dec),
                help="Aging that happens even when idle (not part of net profit).")
    if hybrid:
        c = st.columns(4)
        c[0].metric("Plant revenue (park + battery)", eur(k["Plant revenue"], dec))
        c[1].metric("Park alone revenue", eur(k["Park alone revenue"], dec))
        c[2].metric("Renewable output (MWh)", num(k["Renewable output (MWh)"]))
        c[3].metric("Curtailed (MWh)", num(k["Curtailed (MWh)"]),
                    help=f"Grid charging: {num(k['Grid charging (MWh)'], 'MWh')}")


def compare_table(k_a, k_n, dec, hybrid=True):
    """Side-by-side KPI table for both models."""
    rows = []
    hybrid_only = ("Plant revenue", "Park alone revenue", "Renewable output (MWh)", "Curtailed (MWh)")
    for key in [k for k in k_a if hybrid or k not in hybrid_only]:
        a, n = k_a[key], k_n[key]
        is_money = not any(s in key for s in ("MWh", "%", "cycles", "price"))
        is_price = "price" in key
        f = (lambda v: eur(v, dec)) if is_money else (
            (lambda v: num(v, "€/MWh")) if is_price else (lambda v: num(v, "", 3)))
        diff = a - n if not (pd.isna(a) or pd.isna(n)) else np.nan
        rows.append({"KPI": key, MODEL_LABEL["aware"]: f(a), MODEL_LABEL["naive"]: f(n),
                     "Difference (aware − naive)": f(diff)})
    return pd.DataFrame(rows).set_index("KPI")


def day_table(h, res_row, mode, p):
    """24-hour table for one model and one day."""
    hybrid = mode != "grid"
    dt = config.DT
    t = pd.DataFrame(index=pd.Index(hour_labels(h.index), name="Hour"))
    t["Price (€/MWh)"] = h["Price"].values
    t["Action"] = actions(h)
    if hybrid:
        t[f"{mode.title()} output (MW)"] = h["Gen"].values
    t["Charge from grid (MW)"] = h["P_ch_grid"].values
    if hybrid:
        t[f"Charge from {mode} (MW)"] = h["P_ch_res"].values
    t["Discharge (MW)"] = h["P_dis"].values
    if hybrid:
        t["Sold directly (MW)"] = h["P_res_sell"].values
        t["Curtailed (MW)"] = h["P_curt"].values
    t["SoC (MWh)"] = h["SoC"].values
    e_avail = h["E_avail"].iloc[0]
    t["SoC (%)"] = h["SoC"].values / max(e_avail, 1e-9) * 100

    price = h["Price"].values
    if hybrid:
        plant = price * h["Grid_Export"].values * dt
        g_lim = np.minimum(h["Gen"].values, p["p_grid"]) if p["p_grid"] else h["Gen"].values
        alone = price * np.where(price >= 0, g_lim, 0.0) * dt
        t["Plant revenue (€)"] = plant
        t["Park alone (€)"] = alone
        t["Battery value (€)"] = plant - alone
        value_col = "Battery value (€)"
    else:
        t["Battery cash flow (€)"] = price * (h["P_dis"].values - h["P_ch_grid"].values) * dt
        value_col = "Battery cash flow (€)"

    # Spread the day's health cost over the hours by energy moved
    thr = (h["P_ch"] + h["P_dis"]).values
    share = thr / thr.sum() if thr.sum() > 0 else np.zeros_like(thr)
    t["Health cost (€)"] = share * res_row["health_cost"]
    t["Net (€)"] = t[value_col] - t["Health cost (€)"]

    total = {c: np.nan for c in t.columns}
    for c in t.columns:
        if c.endswith("(€)"):
            total[c] = t[c].sum()
    total["Action"] = "TOTAL"
    t = pd.concat([t, pd.DataFrame([total], index=pd.Index(["Total"], name="Hour"))])
    return t


def style_table(t):
    fmt = {c: "{:,.2f}" for c in t.columns if c != "Action"}
    colors = {"Charge": f"color: {C['charge']}; font-weight: 600",
              "Discharge": f"color: {C['discharge']}; font-weight: 600",
              "Idle": "color: #9ca3af", "TOTAL": "font-weight: 700"}
    return (t.style.format(fmt, na_rep="")
            .map(lambda v: colors.get(v, ""), subset=["Action"]))


# ============================================================
# Charts (Plotly, interactive)
# ============================================================
def day_figure(h, mode, p):
    hybrid = mode != "grid"
    x = hour_labels(h.index)
    act = actions(h)
    bar_colors = [C["charge"] if a == "Charge" else C["discharge"] if a == "Discharge" else C["idle"]
                  for a in act]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.38, 0.37, 0.25],
                        subplot_titles=("Day-ahead price  (bar colour = battery action)",
                                        "Power flows", "State of charge"))
    # Row 1: price bars coloured by action
    fig.add_bar(x=x, y=h["Price"], marker_color=bar_colors, customdata=act, showlegend=False,
                hovertemplate="Price: %{y:,.2f} €/MWh<br>Battery: %{customdata}<extra></extra>",
                row=1, col=1)
    for name, col in (("Charging hour", C["charge"]), ("Discharging hour", C["discharge"]),
                      ("Idle hour", C["idle"])):
        fig.add_scatter(x=[None], y=[None], mode="markers", name=name, legendgroup="price",
                        marker=dict(size=11, symbol="square", color=col), row=1, col=1)
    # Row 2: power flows
    if hybrid:
        fig.add_scatter(x=x, y=h["Gen"], fill="tozeroy", mode="lines", line_shape="hvh",
                        line=dict(color=C["gen"], width=1), fillcolor="rgba(250,204,21,0.35)",
                        name=f"{mode.title()} output",
                        hovertemplate="%{y:,.2f} MW<extra>" + f"{mode.title()} output</extra>",
                        row=2, col=1)
    fig.add_bar(x=x, y=-h["P_ch_grid"], name="Charge from grid", marker_color=C["charge"],
                hovertemplate="%{y:,.2f} MW<extra>Charge from grid</extra>", row=2, col=1)
    if hybrid:
        fig.add_bar(x=x, y=-h["P_ch_res"], name=f"Charge from {mode}", marker_color=C["charge_res"],
                    hovertemplate="%{y:,.2f} MW<extra>" + f"Charge from {mode}</extra>", row=2, col=1)
    fig.add_bar(x=x, y=h["P_dis"], name="Discharge", marker_color=C["discharge"],
                hovertemplate="%{y:,.2f} MW<extra>Discharge</extra>", row=2, col=1)
    # Row 3: SoC with its limits
    e_avail = h["E_avail"].iloc[0]
    fig.add_scatter(x=x, y=h["SoC"], mode="lines+markers", name="State of charge",
                    line=dict(color=C["soc"], width=2),
                    customdata=h["SoC"] / e_avail * 100,
                    hovertemplate="%{y:,.2f} MWh (%{customdata:.1f} %)<extra>SoC</extra>",
                    row=3, col=1)
    for frac, label in ((p["soc_min_frac"], "SoC min"), (p["soc_max_frac"], "SoC max")):
        fig.add_scatter(x=[x[0], x[-1]], y=[frac * e_avail] * 2, mode="lines", showlegend=False,
                        line=dict(color="#9ca3af", dash="dash", width=1),
                        hovertemplate=f"{label}: %{{y:,.2f}} MWh<extra></extra>", row=3, col=1)

    fig.update_layout(height=760, barmode="relative", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
                      margin=dict(t=90, b=40, l=60, r=20))
    fig.update_yaxes(title_text="€/MWh", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="MW  (+ out / − in)", tickformat=",.1f", row=2, col=1)
    fig.update_yaxes(title_text="MWh", tickformat=",.1f", row=3, col=1)
    fig.update_xaxes(title_text="Hour (Berlin time)", row=3, col=1)
    return fig


def range_figures(results, p):
    """Cumulative profit, daily profit and SoH for both models."""
    figs = {}
    fig = go.Figure()
    for m, (res, _) in results.items():
        d = pd.to_datetime(res["date"])
        fig.add_scatter(x=d, y=res["trading_profit"].cumsum(), name=f"{MODEL_LABEL[m]} – battery profit",
                        line=dict(color=C[m]), hovertemplate="%{y:,.0f} €")
        fig.add_scatter(x=d, y=res["net_profit"].cumsum(), name=f"{MODEL_LABEL[m]} – net of health cost",
                        line=dict(color=C[m], dash="dash"), hovertemplate="%{y:,.0f} €")
    fig.update_layout(title="Cumulative profit", yaxis_title="€", yaxis_tickformat=",.0f",
                      hovermode="x unified", height=420, legend=dict(orientation="h", y=-0.2))
    figs["cum"] = fig

    fig = go.Figure()
    for m, (res, _) in results.items():
        fig.add_bar(x=pd.to_datetime(res["date"]), y=res["net_profit"], name=MODEL_LABEL[m],
                    marker_color=C[m], hovertemplate="%{x|%d.%m.%Y}: %{y:,.2f} €")
    fig.update_layout(title="Daily net profit (battery profit − health cost)", barmode="group",
                      yaxis_title="€", yaxis_tickformat=",.0f", height=380, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    figs["daily"] = fig

    fig = go.Figure()
    for m, (res, _) in results.items():
        fig.add_scatter(x=pd.to_datetime(res["date"]), y=res["soh"] * 100, name=MODEL_LABEL[m],
                        line=dict(color=C[m]), hovertemplate="%{y:.4f} %")
    fig.update_layout(title="State of Health", yaxis_title="SoH (%)", yaxis_tickformat=".2f",
                      height=380, hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    figs["soh"] = fig
    return figs


def profile_figure(h, mode, title):
    """Average day: price line + average charging/discharging bars."""
    g = h.groupby(h.index.hour)
    hrs = [f"{x:02d}:00" for x in g["Price"].mean().index]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=hrs, y=-g["P_ch"].mean(), name="Avg charging", marker_color=C["charge"],
                hovertemplate="%{y:,.3f} MW", secondary_y=False)
    fig.add_bar(x=hrs, y=g["P_dis"].mean(), name="Avg discharging", marker_color=C["discharge"],
                hovertemplate="%{y:,.3f} MW", secondary_y=False)
    if mode != "grid":
        fig.add_scatter(x=hrs, y=g["Gen"].mean(), name=f"Avg {mode} output", mode="lines",
                        line=dict(color="#ca8a04", dash="dot"), hovertemplate="%{y:,.2f} MW",
                        secondary_y=False)
    fig.add_scatter(x=hrs, y=g["Price"].mean(), name="Avg price", mode="lines+markers",
                    line=dict(color=C["price"]), hovertemplate="%{y:,.2f} €/MWh", secondary_y=True)
    fig.update_layout(title=title, barmode="relative", hovermode="x unified", height=420,
                      legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="MW (+ discharge / − charge)", tickformat=",.2f", secondary_y=False)
    fig.update_yaxes(title_text="€/MWh", tickformat=",.0f", secondary_y=True)
    return fig


def heatmap_figure(h, title):
    """Net battery power by day x hour (short periods) or month x hour (long periods)."""
    n_days = len(set(h.index.date))
    d = h.assign(hour=h.index.hour)
    if n_days <= 120:
        d["row"] = h.index.strftime("%d.%m.%Y")
        order = list(dict.fromkeys(d["row"]))
    else:
        d["row"] = h.index.strftime("%Y-%m")
        order = sorted(set(d["row"]))
    tab = d.pivot_table(index="row", columns="hour", values="Net_Power", aggfunc="mean").reindex(order)
    vmax = float(np.nanmax(np.abs(tab.values))) or 1.0
    fig = go.Figure(go.Heatmap(
        z=tab.values, x=[f"{c:02d}:00" for c in tab.columns], y=tab.index,
        colorscale="RdYlGn", zmin=-vmax, zmax=vmax,
        colorbar=dict(title="MW"),
        hovertemplate="%{y} %{x}<br>Net power: %{z:,.2f} MW<extra></extra>"))
    fig.update_layout(title=title, height=max(380, min(900, 18 * len(order) + 150)),
                      yaxis=dict(autorange="reversed"))
    return fig


# ============================================================
# PAGE
# ============================================================
st.title("🔋 BESS Dispatch Optimizer")
st.caption("Battery arbitrage on the German day-ahead market (DE-LU) that weighs every trade "
           "against the battery wear it causes.")

tab_in, tab_res, tab_method = st.tabs(["⚙️ Inputs", "📊 Results", "📐 Methodology (Notes & Assumptions)"])

# ============================================================
# TAB 1 - INPUTS
# ============================================================
with tab_in:
    st.subheader("1 · Plant configuration")
    cols = st.columns(3)
    for col, m in zip(cols, config.ASSET_MODES):
        with col:
            img = config.IMAGES_DIR / MODE_IMAGE[m]
            if img.exists():                      # same height for all three pictures
                b64 = base64.b64encode(img.read_bytes()).decode()
                st.markdown(f'<img src="data:image/png;base64,{b64}" '
                            f'style="height:130px;max-width:100%;object-fit:contain">',
                            unsafe_allow_html=True)
            st.markdown(f"**{MODE_LABEL[m]}**")
            st.caption(MODE_TEXT[m])
    mode = st.radio("Select the plant", config.ASSET_MODES, format_func=MODE_LABEL.get,
                    horizontal=True, index=config.ASSET_MODES.index(config.ASSET_MODE))
    hybrid = mode != "grid"

    st.divider()
    st.subheader("2 · Battery")
    c = st.columns(4)
    p_max = c[0].number_input("Power rating (MW)", 0.1, 1000.0, float(config.P_MAX), 0.1, format="%.1f")
    e_nom = c[1].number_input("Energy capacity (MWh)", 0.1, 4000.0, float(config.E_NOM), 0.5, format="%.1f")
    eta_rt = c[2].slider("Round-trip efficiency (%)", 50, 100, int(config.ETA_RT * 100)) / 100
    soh_start = c[3].slider("State of Health at start (%)", 60.0, 100.0, 100.0, 0.5) / 100
    c = st.columns(2)
    soc_lo, soc_hi = c[0].slider("Allowed SoC window (%)", 0, 100,
                                 (int(config.SOC_MIN_FRAC * 100), int(config.SOC_MAX_FRAC * 100)))
    soc_init = c[1].slider("SoC at start and end of each day (%)", 0, 100,
                           int(config.SOC_INIT_FRAC * 100))
    st.caption(f"Duration: **{e_nom / p_max:,.1f} h** (capacity ÷ power) · "
               f"usable energy: **{e_nom * (soc_hi - soc_lo) / 100:,.2f} MWh**")

    st.divider()
    st.subheader("3 · Degradation (battery datasheet)")
    c = st.columns(5)
    capex = c[0].number_input("Replacement cost (€/kWh)", 10.0, 2000.0, float(config.CAPEX_EUR_PER_KWH), 10.0)
    w_a = c[1].number_input("Cycle life at 100 % DoD (a)", 100.0, 50000.0, float(config.WOEHLER_A), 100.0)
    w_b = c[2].number_input("Wöhler exponent (b)", 0.1, 5.0, float(config.WOEHLER_B), 0.05)
    soh_eol = c[3].slider("End-of-Life SoH (%)", 50, 95, int(config.SOH_EOL * 100)) / 100
    cal_life = c[4].number_input("Calendar life (years)", 1.0, 40.0, float(config.CAL_LIFE_YEARS), 1.0)

    st.divider()
    st.subheader("4 · Electricity prices")
    price_src = st.radio("Price data", ["ENTSO-E DE-LU (built-in data)", "Upload my own CSV"],
                         horizontal=True)
    price_upload = None
    if price_src.startswith("ENTSO-E"):
        if CLEAN_FILE.exists():
            _p = cached_base_prices(CLEAN_FILE.stat().st_mtime)
            _days = pd.Series(_p.index.date).drop_duplicates()
            gaps = (_days.diff().dt.days > 1).sum() if len(_days) > 1 else 0
            st.info(f"Saved prices: **{_p.index.min():%d.%m.%Y}** to **{_p.index.max():%d.%m.%Y}** "
                    f"({len(_days):,} days{', with gaps' if gaps else ''}). Day-ahead prices for tomorrow "
                    "are published around 13:00 today. Since 1 Oct 2025 the market uses 15-minute "
                    "prices; they are averaged to hourly here.")
        st.markdown("**Download more prices from ENTSO-E** (merged with the saved data, needs "
                    "`entsoe_api_key.txt`)")
        _first = pd.Timestamp(FIRST_DAY).date()
        _tomorrow = (pd.Timestamp.now(tz=config.LOCAL_TZ) + pd.Timedelta(days=1)).date()
        c = st.columns([1, 1, 1])
        dl_from = c[0].date_input("From", value=_first, min_value=_first, max_value=_tomorrow,
                                  format="DD.MM.YYYY", key="dl_from")
        dl_to = c[1].date_input("To (inclusive)", value=_tomorrow, min_value=_first,
                                max_value=_tomorrow, format="DD.MM.YYYY", key="dl_to")
        c[2].markdown("&nbsp;")
        if c[2].button("⬇️ Download from ENTSO-E", disabled=dl_from > dl_to):
            buf = io.StringIO()
            try:
                with redirect_stdout(buf), st.spinner("Downloading (about 10 s per year) ..."):
                    add_prices(dl_from.isoformat(), (dl_to + timedelta(days=1)).isoformat())
                st.success("Prices downloaded and saved.")
            except Exception as e:                             # e.g. no key, no internet
                st.error(f"Download failed: {e}")
            st.code(buf.getvalue() or "(no output)")
        st.caption("ENTSO-E day-ahead prices start on **5 Jan 2015** (earlier years are not on the "
                   "platform). Until 30 Sep 2018 the bidding zone was DE-AT-LU "
                   "(Germany-Austria-Luxembourg), since 1 Oct 2018 it is DE-LU; both are handled.")
    else:
        price_upload = st.file_uploader("CSV with columns: timestamp, price_eur_mwh", type="csv")
        tmpl = pd.DataFrame({"timestamp": pd.date_range("2025-06-01", periods=48, freq="h"),
                             "price_eur_mwh": np.round(80 + 40 * np.sin(np.arange(48) / 24 * 2 * np.pi), 2)})
        st.download_button("⬇️ Download a template", tmpl.to_csv(index=False).encode(),
                           "price_template.csv", "text/csv")
        st.caption("Times without a timezone are read as Berlin local time. "
                   "15-minute data is averaged to hourly.")

    weather_src, weather_upload, lat, lon = None, None, config.LATITUDE, config.LONGITUDE
    gen_kw, p_grid, allow_grid = {}, None, True
    if hybrid:
        st.divider()
        st.subheader(f"5 · {'Solar' if mode == 'solar' else 'Wind'} park and weather")
        if mode == "solar" and (config.IMAGES_DIR / "solar.png").exists():
            st.image(str(config.IMAGES_DIR / "solar.png"), width=70)
        c = st.columns(4)
        if mode == "solar":
            gen_kw["peak_mw"] = c[0].number_input("PV size (MWp)", 0.1, 2000.0, float(config.PV_PEAK_MW), 0.5)
            gen_kw["pr"] = c[1].slider("Performance ratio", 0.5, 1.0, float(config.PV_PR), 0.01)
        else:
            gen_kw["rated_mw"] = c[0].number_input("Wind park size (MW)", 0.1, 2000.0,
                                                   float(config.WIND_RATED_MW), 0.5)
            gen_kw["hub_height"] = c[1].number_input("Hub height (m)", 20.0, 250.0,
                                                     float(config.WIND_HUB_HEIGHT_M), 5.0)
        p_grid = c[2].number_input("Grid connection (MW)", 0.1, 4000.0, float(config.GRID_CONNECTION_MW), 0.5,
                                   help="Max power the plant may export to (or import from) the grid.")
        allow_grid = c[3].toggle("Battery may also charge from the grid", value=config.ALLOW_GRID_CHARGING,
                                 help="Off = the battery can only store what the park produces "
                                      "(e.g. EEG innovation-tender rules).")
        if mode == "wind":
            c = st.columns(3)
            gen_kw["cut_in"] = c[0].number_input("Cut-in speed (m/s)", 0.0, 10.0, float(config.WIND_CUT_IN), 0.5)
            gen_kw["rated_speed"] = c[1].number_input("Rated speed (m/s)", 5.0, 25.0,
                                                      float(config.WIND_RATED_SPEED), 0.5)
            gen_kw["cut_out"] = c[2].number_input("Cut-out speed (m/s)", 10.0, 40.0, float(config.WIND_CUT_OUT), 0.5)

        weather_src = st.radio("Weather data", ["NASA POWER (latitude / longitude)", "Upload my own CSV"],
                               horizontal=True)
        if weather_src.startswith("NASA"):
            c = st.columns(2)
            lat = c[0].number_input("Latitude", -90.0, 90.0, float(config.LATITUDE), 0.01, format="%.2f")
            lon = c[1].number_input("Longitude", -180.0, 180.0, float(config.LONGITUDE), 0.01, format="%.2f")
            st.caption("NASA POWER is free and needs no key. Its data arrives late: wind about 1 week, "
                       "sunlight about 3 months behind today. Missing hours are filled with a "
                       "**typical-year estimate** (average of the same day and hour in other years), "
                       "and results say when that happened.")
        else:
            weather_upload = st.file_uploader(
                "Hourly CSV: timestamp + one of generation_mw, ghi_w_m2, wind_50m_m_s, wind_hub_m_s",
                type="csv")
            t0 = pd.date_range("2025-06-01", periods=48, freq="h")
            tmpl = pd.DataFrame({"timestamp": t0,
                                 "ghi_w_m2": np.round(np.clip(700 * np.sin((t0.hour - 5) / 15 * np.pi), 0, None), 1),
                                 "wind_50m_m_s": np.round(6 + 2 * np.sin(np.arange(48) / 9), 2)})
            st.download_button("⬇️ Download a template", tmpl.to_csv(index=False).encode(),
                               "weather_template.csv", "text/csv")
            st.caption("generation_mw = park output used directly. ghi_w_m2 = sunlight (converted with "
                       "the PV settings). wind_50m_m_s / wind_hub_m_s = wind speed (converted with the "
                       "turbine settings).")

# Collect all settings for the simulation
params = dict(
    capex_per_kwh=capex, allow_grid_charging=allow_grid, p_grid=p_grid,
    soh_start=soh_start, e_nom=e_nom, woehler_a=w_a, woehler_b=w_b, soh_eol=soh_eol,
    cal_life_years=cal_life, p_max=p_max, eta_rt=eta_rt,
    soc_min_frac=soc_lo / 100, soc_max_frac=soc_hi / 100, soc_init_frac=soc_init / 100,
)
errors = []
if not soc_lo < soc_init < soc_hi:
    errors.append("The start/end SoC must lie inside the allowed SoC window.")
if soh_start <= soh_eol:
    errors.append("The State of Health at start must be above the End-of-Life SoH.")
if soc_hi - soc_lo < 5:
    errors.append("The SoC window is too small.")

dod_ref = (soc_hi - soc_lo) / 100
n_ref = cycle_life(dod_ref, w_a, w_b) if dod_ref > 0 else np.nan
c_deg = marginal_deg_cost(capex, dod_ref, w_a, w_b) if dod_ref > 0 else np.nan
thr = trading_threshold(c_deg, eta_rt)

# ============================================================
# TAB 2 - METHODOLOGY
# ============================================================
with tab_method:
    st.subheader("How the simulator works")
    st.markdown(
        "Every day the optimizer receives the 24 day-ahead prices (23 or 25 on clock-change days) "
        "and decides, hour by hour, how much to **charge**, **discharge** or do nothing. It maximises "
        "market revenue **minus a degradation penalty**, so it only trades when the price spread pays "
        "for the battery wear. After each day the State of Health (SoH) is updated and the next day "
        "starts with the slightly smaller battery.")

    st.markdown("#### 1 · Battery wear: the Wöhler curve")
    st.latex(r"N_{cycle}(DoD) = a \cdot DoD^{-b}")
    st.markdown(f"Cycles until End-of-Life at a depth of discharge *DoD* (a fraction 0–1). With "
                f"a = {w_a:,.0f} and b = {w_b:.2f}: **{cycle_life(1.0, w_a, w_b):,.0f}** cycles at 100 % DoD, "
                f"**{cycle_life(0.5, w_a, w_b):,.0f}** at 50 %, **{n_ref:,.0f}** at the reference DoD "
                f"of {dod_ref:.0%}. Since b > 1, shallow cycles are gentler per MWh than deep ones. "
                "b can be fitted from two datasheet points:")
    st.latex(r"b = \frac{\ln(N_2 / N_1)}{\ln(DoD_1 / DoD_2)}")

    st.markdown("#### 2 · Degradation cost per MWh")
    st.latex(r"Cost_{deg} = \frac{Cost_{capex}}{2 \cdot DoD_{ref} \cdot N_{cycle}(DoD_{ref})}")
    st.markdown(f"Battery cost ÷ lifetime energy moved (in + out, hence the 2). "
                f"{capex:,.0f} €/kWh = {capex * 1000:,.0f} €/MWh → "
                f"{capex * 1000:,.0f} / (2 × {dod_ref:.2f} × {n_ref:,.0f}) = "
                f"**{c_deg:,.2f} €/MWh**, charged on every MWh **charged and discharged**.")

    st.markdown("#### 3 · Trading threshold")
    st.latex(r"\eta_{rt}\cdot\lambda_{dis} - \lambda_{ch} \;>\; Cost_{deg}\cdot(1+\eta_{rt})")
    st.markdown(f"Buying 1 MWh and selling the {eta_rt:.2f} MWh that come back only pays off if "
                f"{eta_rt:.2f} × sell price − buy price > **{thr:,.2f} €/MWh**. Below that the "
                "degradation-aware battery stays idle. The naive model uses Cost_deg = 0 and trades "
                "any spread.")

    st.markdown("#### 4 · Daily optimisation (MILP, solved with HiGHS)")
    st.latex(r"\max \sum_{t} \Big[\lambda_t\,(P_{dis,t} + P_{sell,t} - P_{ch,grid,t})\,\Delta t"
             r" \;-\; Cost_{deg}\,(P_{ch,t} + P_{dis,t})\,\Delta t\Big]")
    st.markdown("**Constraints, for every hour *t*:**")
    st.latex(r"SoC_t = SoC_{t-1} + \eta_{ch} P_{ch,t}\Delta t - \frac{P_{dis,t}\Delta t}{\eta_{dis}},"
             r"\qquad \eta_{ch}=\eta_{dis}=\sqrt{\eta_{rt}}")
    st.latex(r"SoC_{min} \le SoC_t \le SoC_{max}, \qquad SoC_T \ge SoC_{init}")
    st.latex(r"P_{ch,t} \le P_{max}\,u_t,\qquad P_{dis,t} \le P_{max}\,(1-u_t),\qquad u_t \in \{0,1\}")
    st.markdown(f"SoC window {soc_lo}–{soc_hi} % of the **available** capacity "
                f"(= nominal × SoH), start/end {soc_init} %, P_max = {p_max:,.2f} MW. The binary *u* "
                "stops charging and discharging in the same hour, which the solver would otherwise do "
                "at negative prices to waste energy for money.")

    st.markdown("#### 5 · Solar / wind + battery")
    st.latex(r"G_t = P_{sell,t} + P_{ch,res,t} + P_{curt,t},\qquad P_{ch,t} = P_{ch,res,t} + P_{ch,grid,t}")
    st.latex(r"P_{sell,t} + P_{dis,t} \le P_{grid},\qquad P_{ch,grid,t} = 0 \text{ if grid charging is off}")
    st.markdown("The park output *G* goes to the grid, into the battery, or is curtailed (useful at "
                "negative prices). Without grid charging, the battery can only store what the sun or "
                "wind delivers in that hour. **Battery profit** in hybrid mode = plant revenue − "
                "revenue of the park alone (selling everything up to the grid limit, curtailing at "
                "negative prices).")
    st.latex(r"G^{PV}_t = P_{peak}\cdot\frac{GHI_t}{1000\,W/m^2}\cdot PR")
    st.latex(r"v_{hub} = v_{50}\Big(\frac{h_{hub}}{50}\Big)^{0.143},\qquad "
             r"G^{wind}_t = P_{rated}\,\frac{v^3 - v_{in}^3}{v_{rated}^3 - v_{in}^3}"
             r"\;\;(v_{in}\le v < v_{rated})")
    st.markdown("Wind power is 0 below cut-in and above cut-out, and at full power between rated "
                "speed and cut-out.")

    st.markdown("#### 6 · State of Health update (after each day)")
    st.latex(r"\Delta SoH = (1 - SoH_{EoL})\Big[\sum_i \frac{n_i}{N_{cycle}(DoD_i)} + "
             r"\frac{\Delta t_{days}}{L_{cal}}\Big]")
    st.markdown(f"Rainflow counting splits the day's SoC curve into full (n = 1) and half (n = 0.5) "
                f"cycles. The calendar life is {cal_life:,.0f} years and EoL = {soh_eol:.0%}. "
                "**Health cost** = cycling part of ΔSoH ÷ (1 − SoH_EoL) × replacement cost of the "
                "whole battery. The calendar part is shown separately, because it happens even when "
                "the battery is idle. **Net profit** = battery profit − health cost.")

    st.markdown("#### 7 · Assumptions and limitations")
    st.markdown(
        "- **Perfect foresight within each day:** the day-ahead prices are known when the battery "
        "plans (an upper bound; a price forecast would be a later extension).\n"
        "- Hourly resolution. 15-minute prices since Oct 2025 are averaged, which hides some short "
        "spikes.\n"
        "- One constant degradation cost (linearised at the reference DoD); temperature and C-rate "
        "effects are ignored; calendar aging is linear.\n"
        "- Each day starts and ends at the same SoC; no intraday or balancing-market revenue.\n"
        "- The battery may cycle more than once a day, so daily charged / discharged energy can "
        "exceed its capacity (e.g. 2 cycles × 3.2 MWh usable = 6.4 MWh). The SoC itself never "
        "leaves the allowed window. Charged energy is measured at the grid side, before losses.\n"
        "- PV and wind models are simple (flat-plate irradiance, generic power curve); NASA gaps are "
        "filled with typical-year values.")

# ============================================================
# TAB 3 - RESULTS
# ============================================================
with tab_res:
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # ---- Prices ----
    if price_src.startswith("ENTSO-E"):
        if not CLEAN_FILE.exists():
            st.warning("No built-in price data yet. Run `python src/data_loader.py` or upload a CSV.")
            st.stop()
        prices = cached_base_prices(CLEAN_FILE.stat().st_mtime)
    else:
        if price_upload is None:
            st.info("⬆️ Upload a price CSV in the **Inputs** tab.")
            st.stop()
        try:
            prices = cached_upload_prices(price_upload.getvalue())
        except Exception as e:
            st.error(f"Could not read the price file: {e}")
            st.stop()

    # ---- Renewable generation (solar / wind) ----
    gen, gen_filled = None, None
    if hybrid:
        try:
            if weather_src.startswith("NASA"):
                start = min(prices.index.min().tz_convert("UTC").strftime("%Y-%m-%d"), config.START)
                w = cached_nasa(round(lat, 2), round(lon, 2), start)
                is_upload = False
            else:
                if weather_upload is None:
                    st.info("⬆️ Upload a weather CSV in the **Inputs** tab.")
                    st.stop()
                w = cached_weather_upload(weather_upload.getvalue())
                is_upload = True
            g = cached_generation(prices, mode, w, is_upload, tuple(sorted(gen_kw.items())))
            gen, gen_filled = g["gen_mw"], g["filled"]
        except Exception as e:
            st.error(f"Weather data problem: {e}")
            st.stop()

    first_day, last_day = prices.index.min().date(), prices.index.max().date()

    # Default date: the latest day, or for solar/wind the latest day with MEASURED weather
    default_day, weather_note = last_day, ""
    if gen_filled is not None:
        measured = ~gen_filled.groupby(gen_filled.index.date).any()
        measured_days = measured[measured].index
        if len(measured_days):
            default_day = max(measured_days)
        src = "NASA POWER publishes sunlight about 3 months late and wind about 1 week late." \
            if weather_src.startswith("NASA") else "The uploaded weather file does not cover every hour."
        weather_note = (f"{src} Last day with measured weather: **{default_day:%d.%m.%Y}**."
                        if len(measured_days) else src)
        if default_day < last_day:
            st.caption(f"🌤️ {weather_note} Later days use a typical-year weather estimate.")

    st.markdown(f"**{MODE_LABEL[mode]}** · {p_max:,.1f} MW / {e_nom:,.1f} MWh · degradation cost "
                f"**{c_deg:,.2f} €/MWh** · trading threshold **{thr:,.2f} €/MWh** · data "
                f"{first_day:%d.%m.%Y} – {last_day:%d.%m.%Y}")
    view = st.radio("View", ["📅 Single day", "📆 Date range"], horizontal=True,
                    label_visibility="collapsed")
    p_items = tuple(sorted(params.items()))

    # -------------------- SINGLE DAY --------------------
    if view.startswith("📅"):
        day = st.date_input("Select a day", value=default_day, min_value=first_day, max_value=last_day,
                            format="DD.MM.YYYY")
        p_day = prices[prices.index.date == day]
        if p_day.empty:
            st.warning("No prices for this day.")
            st.stop()
        g_day = None if gen is None else gen.loc[p_day.index]
        if gen_filled is not None and gen_filled.loc[p_day.index].any():
            st.info(f"ℹ️ {weather_note} For this day, {gen_filled.loc[p_day.index].mean():.0%} of the "
                    "hours use the **typical-year estimate** instead of measured weather.")

        with st.spinner("Optimising ..."):
            out = {m: cached_sim(p_day, g_day, m == "naive", mode, p_items) for m in ("aware", "naive")}
        k = {m: kpis(out[m][0], out[m][1], params) for m in out}

        t_a, t_n, t_c = st.tabs([MODEL_LABEL["aware"], MODEL_LABEL["naive"], "⚖️ Comparison"])
        for m, tab in (("aware", t_a), ("naive", t_n)):
            with tab:
                res, h = out[m]
                show_kpis(k[m], hybrid, 2)
                st.markdown(f"##### {day:%A, %d.%m.%Y} – {MODEL_LABEL[m]}")
                st.plotly_chart(day_figure(h, mode, params), width="stretch", key=f"day_fig_{m}")
                st.markdown("##### 24-hour schedule")
                tbl = day_table(h, res.iloc[0], mode, params)
                st.dataframe(style_table(tbl), width="stretch", height=920)
                st.download_button("⬇️ Download this table (CSV)", tbl.to_csv().encode(),
                                   f"schedule_{mode}_{m}_{day}.csv", "text/csv", key=f"dl_day_{m}")
        with t_c:
            _ct = compare_table(k["aware"], k["naive"], 2, hybrid)
            st.dataframe(_ct, width="stretch", height=35 * (len(_ct) + 1) + 3)
            fig = go.Figure()
            for m in ("aware", "naive"):
                h = out[m][1]
                fig.add_scatter(x=hour_labels(h.index), y=h["SoC"], name=MODEL_LABEL[m],
                                mode="lines+markers", line=dict(color=C[m]),
                                hovertemplate="%{y:,.2f} MWh")
            fig.update_layout(title="State of charge – both models", yaxis_title="MWh",
                              hovermode="x unified", height=380)
            st.plotly_chart(fig, width="stretch", key="day_fig_compare")

    # -------------------- DATE RANGE --------------------
    else:
        default_start = max(first_day, default_day - timedelta(days=30))
        rng = st.date_input("Select a period", value=(default_start, default_day), min_value=first_day,
                            max_value=last_day, format="DD.MM.YYYY")
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            st.info("Pick a start **and** an end date.")
            st.stop()
        d0, d1 = rng
        mask = (prices.index.date >= d0) & (prices.index.date <= d1)
        p_rng = prices[mask]
        n_days = len(set(p_rng.index.date))
        st.caption(f"{n_days:,} days · roughly {n_days * 0.15:,.0f}–{n_days * 0.3:,.0f} s for both models. "
                   "The battery starts with the SoH from the Inputs tab and ages day by day.")
        key = (mode, d0, d1, p_items, price_src, None if price_upload is None else hashlib.md5(price_upload.getvalue()).hexdigest(),
               weather_src, None if weather_upload is None else hashlib.md5(weather_upload.getvalue()).hexdigest(),
               lat, lon, tuple(sorted(gen_kw.items())))

        if st.button("▶️ Run simulation", type="primary"):
            g_rng = None if gen is None else gen.loc[p_rng.index]
            results = {}
            for m in ("aware", "naive"):
                bar = st.progress(0.0, text=f"{MODEL_LABEL[m]} ...")
                results[m] = run_sim(p_rng, g_rng, m == "naive", mode, params,
                                     progress=lambda f, b=bar, mm=m: b.progress(
                                         min(f, 1.0), text=f"{MODEL_LABEL[mm]}: {f:.0%}"))
                bar.empty()
            st.session_state["range"] = (key, results)

        saved = st.session_state.get("range")
        if saved is None or saved[0] != key:
            st.info("Press **Run simulation** to calculate this period with the current inputs.")
            st.stop()
        results = saved[1]
        if gen_filled is not None and gen_filled.loc[p_rng.index].any():
            st.info(f"ℹ️ {weather_note} In this period, {gen_filled.loc[p_rng.index].mean():.0%} of the "
                    "hours use the **typical-year estimate** instead of measured weather.")
        if len(results["aware"][0]) < n_days or len(results["naive"][0]) < n_days:
            st.warning("A battery reached End-of-Life before the end of the period; its simulation stopped there.")

        k = {m: kpis(results[m][0], results[m][1], params) for m in results}
        figs = range_figures(results, params)
        t_a, t_n, t_c = st.tabs([MODEL_LABEL["aware"], MODEL_LABEL["naive"], "⚖️ Comparison"])
        for m, tab in (("aware", t_a), ("naive", t_n)):
            with tab:
                res, h = results[m]
                show_kpis(k[m], hybrid, 0)
                c = st.columns(2)
                c[0].plotly_chart(profile_figure(h, mode, "Average day – when does it trade?"),
                                  width="stretch", key=f"rng_profile_{m}")
                c[1].plotly_chart(heatmap_figure(h, "Net battery power (red = charge, green = discharge)"),
                                  width="stretch", key=f"rng_heat_{m}")
                st.markdown("##### Daily results")
                daily = pd.DataFrame({
                    "Date": pd.to_datetime(res["date"]).dt.strftime("%d.%m.%Y"),
                    "Battery profit (€)": res["trading_profit"],
                    "Health cost (€)": res["health_cost"],
                    "Net profit (€)": res["net_profit"],
                    "Charged (MWh)": res["charged_mwh"],
                    "Discharged (MWh)": res["discharged_mwh"],
                    "Avg charge price (€/MWh)": res["avg_charge_price"],
                    "Avg discharge price (€/MWh)": res["avg_discharge_price"],
                    "Min price (€/MWh)": res["price_min"],
                    "Max price (€/MWh)": res["price_max"],
                    "SoH (%)": res["soh"] * 100,
                })
                if hybrid:
                    daily.insert(4, "Renewable output (MWh)", res["gen_mwh"])
                    daily.insert(5, "Curtailed (MWh)", res["curtailed_mwh"])
                daily = daily.set_index("Date")
                fmt = {c2: "{:,.2f}" for c2 in daily.columns}
                fmt["SoH (%)"] = "{:,.4f}"
                st.dataframe(daily.style.format(fmt, na_rep="–"), width="stretch", height=420)
                c = st.columns(2)
                c[0].download_button("⬇️ Daily results (CSV)", res.to_csv(index=False).encode(),
                                     f"daily_{mode}_{m}_{d0}_{d1}.csv", "text/csv", key=f"dl_d_{m}")
                c[1].download_button("⬇️ All hours (CSV)", h.to_csv(index_label="timestamp").encode(),
                                     f"hourly_{mode}_{m}_{d0}_{d1}.csv", "text/csv", key=f"dl_h_{m}")
        with t_c:
            _ct = compare_table(k["aware"], k["naive"], 0, hybrid)
            st.dataframe(_ct, width="stretch", height=35 * (len(_ct) + 1) + 3)
            st.plotly_chart(figs["cum"], width="stretch", key="rng_cum")
            st.plotly_chart(figs["daily"], width="stretch", key="rng_daily")
            st.plotly_chart(figs["soh"], width="stretch", key="rng_soh")
            both = pd.concat({m: results[m][0] for m in results}, names=["model"]).reset_index(level=0)
            st.download_button("⬇️ Both models, daily (CSV)", both.to_csv(index=False).encode(),
                               f"daily_{mode}_both_{d0}_{d1}.csv", "text/csv", key="dl_both")
