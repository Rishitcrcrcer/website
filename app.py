"""
HMM Regime Trading Strategy — Interactive Dashboard
=====================================================
Run with:  streamlit run app.py

Expects (in the same folder, or uploaded via the sidebar):
    - hmm_oos_results.csv   (index=date, must contain a 'regime' column with Bull/Chop/Bear)
    - price_data.csv        (yfinance-style export, index=date, has a close/price column)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="HMM Regime Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REGIME_COLORS = {"Bull": "#2ECC71", "Chop": "#F4D03F", "Bear": "#E74C3C"}
STRAT_COLORS = {
    "aggressive": "#E74C3C",
    "balanced": "#3498DB",
    "conservative": "#2ECC71",
    "advanced_max": "#9B59B6",
    "advanced_safest": "#F39C12",
    "custom": "#1ABC9C",
}

BEST_STRATEGIES = {
    "aggressive": {
        "bull": 2.3, "chop": 1.5, "bear": 0.5,
        "label": "🚀 Aggressive — Max Return",
        "use_drawdown_penalty": False, "use_volatility_scaling": False,
    },
    "balanced": {
        "bull": 1.0, "chop": 0.7, "bear": 0.3,
        "label": "⚖️ Balanced — Best Balance",
        "use_drawdown_penalty": False, "use_volatility_scaling": False,
    },
    "conservative": {
        "bull": 1.0, "chop": 0.3, "bear": 0.0,
        "label": "🛡️ Conservative — Safest",
        "use_drawdown_penalty": False, "use_volatility_scaling": False,
    },
    "advanced_max": {
        "bull": 2.5, "chop": 2.0, "bear": 1.0,
        "label": "🔥 Advanced Max — Risk-Managed Max Return",
        "use_drawdown_penalty": True, "use_volatility_scaling": True,
    },
    "advanced_safest": {
        "bull": 1.0, "chop": 0.5, "bear": 0.2,
        "label": "🏆 Advanced Safest — Best Sharpe",
        "use_drawdown_penalty": True, "use_volatility_scaling": True,
    },
}

TRANSACTION_COST = 0.0002

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data
def load_data(oos_file, price_file):
    oos_df = pd.read_csv(oos_file, index_col=0, parse_dates=True)

    # If the OOS file already has a close/price column, drop it so the join
    # doesn't collide with price_df's close column.
    existing_close_cols = [c for c in oos_df.columns if c.lower() in ("close", "price")]
    if existing_close_cols:
        oos_df = oos_df.drop(columns=existing_close_cols)

    price_df = pd.read_csv(price_file, skiprows=[1, 2], index_col=0, parse_dates=True)
    price_df.columns = price_df.columns.str.strip().str.lower()
    close_col = [c for c in price_df.columns if "close" in c or "price" in c][0]
    price_df = price_df[[close_col]].rename(columns={close_col: "close"})

    oos_df = oos_df.join(price_df, how="left")
    oos_df["close"] = oos_df["close"].ffill()
    oos_df["asset_return"] = oos_df["close"].pct_change()
    return oos_df


# ------------------------------------------------------------------
# Backtest engine (vectorised-ish, same logic as original script)
# ------------------------------------------------------------------
def calculate_position(regime, current_drawdown, volatility, params):
    if regime == "Bull":
        base = params["bull"]
    elif regime == "Chop":
        base = params["chop"]
    else:
        base = params["bear"]

    if params.get("use_drawdown_penalty", False):
        if current_drawdown < -0.10:
            penalty = max(1.0 + current_drawdown * 2, 0.3)
        elif current_drawdown < -0.05:
            penalty = max(1.0 - (abs(current_drawdown) - 0.05) * 5, 0.5)
        else:
            penalty = 1.0
        base *= penalty

    if params.get("use_volatility_scaling", False):
        target_vol = 0.15
        vol_scalar = min(1.0, target_vol / max(volatility, 0.01))
        base *= vol_scalar

    return float(np.clip(base, -2.0, 3.0))


@st.cache_data(show_spinner=False)
def run_backtest(oos_df, params_tuple):
    params = dict(params_tuple)
    df = oos_df.copy()
    df["position"] = 0.0

    df["volatility"] = df["asset_return"].rolling(20).std() * np.sqrt(252)
    df["volatility"] = df["volatility"].fillna(0.15)

    cum_returns = pd.Series(1.0, index=df.index)
    positions = np.zeros(len(df))

    regimes = df["regime"].values
    vols = df["volatility"].values
    asset_returns = df["asset_return"].values
    cum_vals = np.ones(len(df))
    running_max = 1.0

    for i in range(1, len(df)):
        current_regime = regimes[i - 1]
        current_volatility = vols[i - 1]
        current_cum = cum_vals[i - 1]
        running_max = max(running_max, current_cum)
        current_dd = (current_cum - running_max) / running_max if running_max > 0 else 0

        pos = calculate_position(current_regime, current_dd, current_volatility, params)

        ret = pos * asset_returns[i]
        cum_vals[i] = cum_vals[i - 1] * (1 + ret)
        positions[i] = pos

    df["position"] = positions
    df["strategy_return"] = df["position"].shift(1) * df["asset_return"]
    df["turnover"] = df["position"].diff().abs()
    df["transaction_cost"] = df["turnover"] * TRANSACTION_COST
    df["net_return"] = df["strategy_return"] - df["transaction_cost"]
    df["cum_strategy"] = (1 + df["net_return"]).cumprod()
    df["cum_buyhold"] = (1 + df["asset_return"]).cumprod()

    return df


def calculate_metrics(df):
    clean = df.dropna(subset=["net_return"])

    total_strategy = (clean["cum_strategy"].iloc[-1] - 1) * 100
    total_buyhold = (clean["cum_buyhold"].iloc[-1] - 1) * 100

    years = len(clean) / 252
    ann_strategy = ((1 + total_strategy / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

    vol_strategy = clean["net_return"].std() * np.sqrt(252) * 100
    sharpe = ann_strategy / vol_strategy if vol_strategy > 0 else 0

    cum = clean["cum_strategy"]
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min() * 100

    wins = (clean["net_return"] > 0).sum()
    active_days = (clean["position"] != 0).sum()
    win_rate = wins / active_days * 100 if active_days > 0 else 0

    calmar = ann_strategy / abs(max_dd) if max_dd != 0 else 0

    avg_position = clean["position"].abs().mean()
    max_position = clean["position"].abs().max()
    avg_turnover = clean["turnover"].mean() * 100

    return {
        "total_return": total_strategy,
        "annual_return": ann_strategy,
        "volatility": vol_strategy,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "calmar": calmar,
        "avg_position": avg_position,
        "max_position": max_position,
        "avg_turnover": avg_turnover,
        "excess_return": total_strategy - total_buyhold,
        "buyhold_return": total_buyhold,
    }


def drawdown_series(cum):
    running_max = cum.expanding().max()
    return (cum - running_max) / running_max * 100


# ------------------------------------------------------------------
# Sidebar — data + strategy selection
# ------------------------------------------------------------------
st.sidebar.title("📈 HMM Strategy Dashboard")
st.sidebar.markdown("---")

default_oos = "hmm_oos_results.csv"
default_price = "price_data.csv"

try:
    oos_df = load_data(default_oos, default_price)
    data_ok = True
except Exception as e:
    data_ok = False
    st.sidebar.error(f"Could not load data: {e}")
    st.error(
        "⚠️ Could not find/read the data files.\n\n"
        "Make sure **hmm_oos_results.csv** and **price_data.csv** are placed in the "
        "same folder as this app."
    )
    st.stop()

st.sidebar.success(f"✅ Loaded {len(oos_df)} rows\n{oos_df.index[0].date()} → {oos_df.index[-1].date()}")

st.sidebar.markdown("---")
st.sidebar.subheader("1️⃣ Choose Strategy")

strategy_names = list(BEST_STRATEGIES.keys()) + ["custom"]
strategy_labels = [BEST_STRATEGIES[s]["label"] for s in BEST_STRATEGIES] + ["🎛️ Custom — Build Your Own"]

choice_idx = st.sidebar.radio(
    "Strategy",
    options=list(range(len(strategy_names))),
    format_func=lambda i: strategy_labels[i],
    index=0,
)
selected_strategy = strategy_names[choice_idx]

st.sidebar.markdown("---")
st.sidebar.subheader("2️⃣ Compare Strategies")
compare_mode = st.sidebar.checkbox("Show comparison across ALL strategies", value=False)

# Build params for selected strategy (with custom override)
if selected_strategy == "custom":
    st.sidebar.markdown("#### 🎛️ Custom Parameters")
    bull = st.sidebar.slider("Bull exposure", -2.0, 3.0, 1.0, 0.1)
    chop = st.sidebar.slider("Chop exposure", -2.0, 3.0, 0.7, 0.1)
    bear = st.sidebar.slider("Bear exposure", -2.0, 3.0, 0.3, 0.1)
    use_dd = st.sidebar.checkbox("Enable drawdown penalty", value=False)
    use_vol = st.sidebar.checkbox("Enable volatility scaling", value=False)
    params = {
        "bull": bull, "chop": chop, "bear": bear,
        "use_drawdown_penalty": use_dd, "use_volatility_scaling": use_vol,
    }
else:
    base_params = BEST_STRATEGIES[selected_strategy]
    st.sidebar.markdown("#### 🔧 Fine-tune (optional)")
    tune = st.sidebar.checkbox("Adjust parameters", value=False)
    if tune:
        bull = st.sidebar.slider("Bull exposure", -2.0, 3.0, float(base_params["bull"]), 0.1)
        chop = st.sidebar.slider("Chop exposure", -2.0, 3.0, float(base_params["chop"]), 0.1)
        bear = st.sidebar.slider("Bear exposure", -2.0, 3.0, float(base_params["bear"]), 0.1)
        use_dd = st.sidebar.checkbox("Enable drawdown penalty", value=base_params["use_drawdown_penalty"])
        use_vol = st.sidebar.checkbox("Enable volatility scaling", value=base_params["use_volatility_scaling"])
    else:
        bull, chop, bear = base_params["bull"], base_params["chop"], base_params["bear"]
        use_dd, use_vol = base_params["use_drawdown_penalty"], base_params["use_volatility_scaling"]
    params = {
        "bull": bull, "chop": chop, "bear": bear,
        "use_drawdown_penalty": use_dd, "use_volatility_scaling": use_vol,
    }

# ------------------------------------------------------------------
# Run backtest(s)
# ------------------------------------------------------------------
params_tuple = tuple(sorted(params.items()))
result_df = run_backtest(oos_df, params_tuple)
metrics = calculate_metrics(result_df)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
title_label = params.get("label") if "label" in params else (
    BEST_STRATEGIES[selected_strategy]["label"] if selected_strategy != "custom" else "🎛️ Custom Strategy"
)
st.title("📊 HMM Regime Trading Strategy Dashboard")
st.caption(f"Currently viewing: **{title_label}**  |  Bull `{params['bull']}` · Chop `{params['chop']}` · Bear `{params['bear']}`"
           f"  |  DD-penalty `{params['use_drawdown_penalty']}` · Vol-scaling `{params['use_volatility_scaling']}`")

# ------------------------------------------------------------------
# KPI row
# ------------------------------------------------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Return", f"{metrics['total_return']:.1f}%", f"{metrics['excess_return']:+.1f}% vs B&H")
c2.metric("Annualized Return", f"{metrics['annual_return']:.2f}%")
c3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
c4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
c5.metric("Calmar Ratio", f"{metrics['calmar']:.2f}")
c6.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

st.markdown("---")

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Price & Regimes", "💰 Performance", "📉 Drawdown", "🔁 Transition Matrix", "📋 Compare All"]
)

# ---- TAB 1: Price with regime shading ----
with tab1:
    st.subheader("SPY Price History with HMM Detected Regimes")

    fig = go.Figure()

    # shade regimes as background rectangles (group consecutive same-regime runs)
    regime_series = result_df["regime"]
    change_points = regime_series.ne(regime_series.shift()).cumsum()
    for _, grp in result_df.groupby(change_points):
        r = grp["regime"].iloc[0]
        fig.add_vrect(
            x0=grp.index[0], x1=grp.index[-1],
            fillcolor=REGIME_COLORS.get(r, "#CCCCCC"), opacity=0.25, line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=result_df.index, y=result_df["close"], mode="lines",
        line=dict(color="black", width=1.3), name="Close Price"
    ))

    for r, c in REGIME_COLORS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                  marker=dict(size=10, color=c), name=f"{r} Regime"))

    fig.update_layout(height=520, hovermode="x unified",
                       yaxis_title="Price (USD)", xaxis_title="Date",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Regime distribution"):
        counts = result_df["regime"].value_counts()
        pie = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values,
                                      marker=dict(colors=[REGIME_COLORS.get(x, "#999") for x in counts.index]),
                                      hole=0.4)])
        pie.update_layout(height=350)
        st.plotly_chart(pie, use_container_width=True)

# ---- TAB 2: Performance (cumulative returns + position) ----
with tab2:
    st.subheader("Cumulative Returns: Strategy vs Buy & Hold")

    fig2 = make_subplots(specs=[[{"secondary_y": False}]])
    fig2.add_trace(go.Scatter(
        x=result_df.index, y=result_df["cum_strategy"], mode="lines",
        name=title_label, line=dict(color=STRAT_COLORS.get(selected_strategy, "#1ABC9C"), width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=result_df.index, y=result_df["cum_buyhold"], mode="lines",
        name="Buy & Hold", line=dict(color="#7F8C8D", width=1.5, dash="dash")
    ))
    fig2.update_layout(height=480, hovermode="x unified",
                        yaxis_title="Cumulative Return (x)", xaxis_title="Date",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Position Sizing Over Time")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=result_df.index, y=result_df["position"], mode="lines",
        line=dict(color="#2980B9", width=1), fill="tozeroy", name="Position"
    ))
    fig3.add_hline(y=0, line_color="black", line_width=1)
    fig3.update_layout(height=320, yaxis_title="Position Size (leverage)", xaxis_title="Date")
    st.plotly_chart(fig3, use_container_width=True)

# ---- TAB 3: Drawdown comparison ----
with tab3:
    st.subheader("Drawdown Comparison: Strategy vs Buy & Hold")

    dd_strategy = drawdown_series(result_df["cum_strategy"])
    dd_bh = drawdown_series(result_df["cum_buyhold"])

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=result_df.index, y=dd_strategy, mode="lines",
                               name="Strategy", line=dict(color="#2ECC71"),
                               fill="tozeroy"))
    fig4.add_trace(go.Scatter(x=result_df.index, y=dd_bh, mode="lines",
                               name="Buy & Hold", line=dict(color="#E74C3C"),
                               fill="tozeroy"))
    fig4.add_hline(y=dd_strategy.min(), line_dash="dot", line_color="#2ECC71",
                   annotation_text=f"Strategy Max DD: {dd_strategy.min():.2f}%")
    fig4.add_hline(y=dd_bh.min(), line_dash="dot", line_color="#E74C3C",
                   annotation_text=f"B&H Max DD: {dd_bh.min():.2f}%")
    fig4.update_layout(height=500, hovermode="x unified",
                        yaxis_title="Drawdown (%)", xaxis_title="Date",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig4, use_container_width=True)

    dcol1, dcol2 = st.columns(2)
    dcol1.metric("Strategy Max Drawdown", f"{dd_strategy.min():.2f}%")
    dcol2.metric("Buy & Hold Max Drawdown", f"{dd_bh.min():.2f}%")

# ---- TAB 4: Transition matrix ----
with tab4:
    st.subheader("HMM Regime Transition Probability Matrix")

    regimes_order = ["Bull", "Chop", "Bear"]
    reg = result_df["regime"].values
    trans = pd.DataFrame(0, index=regimes_order, columns=regimes_order, dtype=float)
    for i in range(len(reg) - 1):
        a, b = reg[i], reg[i + 1]
        if a in regimes_order and b in regimes_order:
            trans.loc[a, b] += 1
    trans = trans.div(trans.sum(axis=1), axis=0).fillna(0)

    fig5 = go.Figure(data=go.Heatmap(
        z=trans.values, x=[f"To {c}" for c in trans.columns], y=[f"From {r}" for r in trans.index],
        colorscale=[[0, "#A93226"], [0.5, "#F4D03F"], [1, "#186A3B"]],
        text=[[f"{v:.3f}" for v in row] for row in trans.values],
        texttemplate="%{text}", textfont=dict(size=18, color="white"),
        zmin=0, zmax=1, showscale=True,
    ))
    fig5.update_layout(height=450, xaxis_title="To Regime", yaxis_title="From Regime")
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Diagonal values = regime persistence probability. High diagonal = stable regime, low diagonal = frequent switching.")

# ---- TAB 5: Compare all strategies ----
with tab5:
    st.subheader("Strategy Comparison")

    if compare_mode or st.button("🔄 Run comparison across all 5 strategies"):
        rows = []
        cum_curves = {}
        for name, p in BEST_STRATEGIES.items():
            pt = tuple(sorted({k: v for k, v in p.items() if k in
                                ["bull", "chop", "bear", "use_drawdown_penalty", "use_volatility_scaling"]}.items()))
            df_i = run_backtest(oos_df, pt)
            m = calculate_metrics(df_i)
            m["strategy"] = name
            rows.append(m)
            cum_curves[name] = df_i["cum_strategy"]

        comp_df = pd.DataFrame(rows).set_index("strategy")

        st.markdown("#### 📋 Metrics Table")
        display_cols = ["total_return", "annual_return", "volatility", "sharpe",
                         "max_drawdown", "win_rate", "calmar", "excess_return"]
        st.dataframe(
            comp_df[display_cols].style.format("{:.2f}"),
            use_container_width=True,
        )

        st.markdown("#### 📈 Cumulative Return Comparison")
        fig6 = go.Figure()
        for name, curve in cum_curves.items():
            fig6.add_trace(go.Scatter(x=curve.index, y=curve, mode="lines",
                                       name=name, line=dict(color=STRAT_COLORS.get(name))))
        fig6.add_trace(go.Scatter(x=result_df.index, y=result_df["cum_buyhold"], mode="lines",
                                   name="Buy & Hold", line=dict(color="black", dash="dot")))
        fig6.update_layout(height=500, hovermode="x unified", yaxis_title="Cumulative Return (x)")
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("#### 📊 Bar Chart Comparison")
        metric_choice = st.selectbox(
            "Metric to compare",
            ["total_return", "sharpe", "max_drawdown", "calmar", "win_rate", "annual_return"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        fig7 = go.Figure(go.Bar(
            x=comp_df.index, y=comp_df[metric_choice],
            marker_color=[STRAT_COLORS.get(i, "#999") for i in comp_df.index],
            text=comp_df[metric_choice].round(2), textposition="outside",
        ))
        fig7.update_layout(height=420, yaxis_title=metric_choice.replace("_", " ").title())
        st.plotly_chart(fig7, use_container_width=True)

        best_return = comp_df["total_return"].idxmax()
        best_sharpe = comp_df["sharpe"].idxmax()
        best_dd = comp_df["max_drawdown"].idxmax()
        best_calmar = comp_df["calmar"].idxmax()

        st.markdown("#### 🏆 Recommendations")
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.info(f"**Max Return**\n\n{best_return}\n\n{comp_df.loc[best_return,'total_return']:.1f}%")
        rcol2.success(f"**Best Sharpe**\n\n{best_sharpe}\n\n{comp_df.loc[best_sharpe,'sharpe']:.1f}")
        rcol3.warning(f"**Lowest Drawdown**\n\n{best_dd}\n\n{comp_df.loc[best_dd,'max_drawdown']:.1f}%")
        rcol4.info(f"**Best Calmar**\n\n{best_calmar}\n\n{comp_df.loc[best_calmar,'calmar']:.1f}")
    else:
        st.info("Check 'Show comparison across ALL strategies' in the sidebar, or click the button above, to compare all 5 strategies side by side.")

st.markdown("---")
st.caption("Built from your HMM regime-detection backtest logic • Position sizing, drawdown penalty & volatility scaling match the original script exactly.")
