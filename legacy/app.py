"""Streamlit inference app for stock_future.

Loads the trained model from ``artifacts/model.pt`` (plus the feature scaler
from ``artifacts/scaler.npz``), lets the user pick any TSE Prime ticker, and
forecasts the next 20 trading days of price movement using the most recent
60-day window of features.

Run::

    /c/Users/matsu/anaconda3/python.exe -m streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yfinance as yf

from prepare import (
    ARTIFACTS_DIR,
    FEATURE_COLS,
    HORIZON,
    LOOKBACK,
    build_features,
    FeatureScaler,
)
from train import LSTMForecaster
from universe import UNIVERSE


CKPT_PATH = ARTIFACTS_DIR / "model.pt"
SCALER_PATH = ARTIFACTS_DIR / "scaler.npz"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> tuple[torch.nn.Module, dict, FeatureScaler]:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = LSTMForecaster(
        in_features=cfg["in_features"],
        hidden=cfg["hidden"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        horizon=cfg["horizon"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = FeatureScaler.load(SCALER_PATH)
    return model, ckpt, scaler


@st.cache_data(ttl=60 * 30)
def fetch_history(ticker: str, end_date: datetime | None = None,
                  trading_days: int = 180) -> pd.DataFrame | None:
    if end_date is None:
        end_date = datetime.now()
    start = end_date - timedelta(days=trading_days * 2 + 30)  # buffer for weekends/holidays
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        st.error(f"yfinance の取得に失敗しました: {e}")
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return df


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_next_returns(model: torch.nn.Module, scaler: FeatureScaler,
                         df: pd.DataFrame) -> np.ndarray | None:
    """Return a [HORIZON] np.ndarray of predicted daily log returns."""
    feats = build_features(df)
    feats = feats.dropna(how="any")
    if len(feats) < LOOKBACK:
        return None
    window = feats.iloc[-LOOKBACK:].values.astype(np.float32)
    window = scaler.transform(window[None, ...])  # [1, LOOKBACK, F]
    with torch.no_grad():
        out = model(torch.from_numpy(window))
    pred = out.squeeze(0).numpy()
    if pred.ndim == 0:
        # Scalar (cumulative) model — distribute evenly across HORIZON
        pred = np.full(HORIZON, float(pred) / HORIZON, dtype=np.float32)
    elif pred.shape[0] != HORIZON:
        pred = np.resize(pred, HORIZON)
    return pred


def build_forecast_path(last_close: float, pred_daily_logrets: np.ndarray,
                        start_date: pd.Timestamp) -> pd.DataFrame:
    cum = np.cumsum(pred_daily_logrets)
    prices = last_close * np.exp(cum)
    dates = pd.bdate_range(start=start_date + pd.offsets.BDay(1),
                           periods=len(prices))
    return pd.DataFrame({"Date": dates, "ForecastClose": prices}).set_index("Date")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="stock_future — 1 か月予測", layout="wide")
st.title("📈 stock_future — 東証プライム株の 20 営業日先予測")
st.caption("karpathy/autoresearch 流の自己改善ループで鍛えた LSTM モデルによる予測 "
           "(研究目的。投資助言ではありません)")

# --- Sidebar ---
with st.sidebar:
    st.header("銘柄選択")
    universe_labels = [f"{t}  {n}  [{s}]" for t, n, s in UNIVERSE]
    idx_default = 0
    for i, (t, _n, _s) in enumerate(UNIVERSE):
        if t == "7203.T":
            idx_default = i
            break
    choice = st.selectbox(
        "ユニバースから選ぶ",
        options=list(range(len(UNIVERSE))),
        format_func=lambda i: universe_labels[i],
        index=idx_default,
    )
    ticker_default = UNIVERSE[choice][0]
    ticker = st.text_input("またはティッカー (例 7203.T, 6758.T)",
                           value=ticker_default).strip()
    if ticker and not ticker.endswith(".T") and ticker.isdigit():
        ticker = f"{ticker}.T"
    base_date = st.date_input("予測基準日", value=datetime.now().date())
    go_btn = st.button("予測を実行", type="primary", use_container_width=True)


# --- Load model once ---
if not CKPT_PATH.exists():
    st.error(f"学習済みモデルが見つかりません: {CKPT_PATH}\n"
             "先に `python train.py` を実行してください。")
    st.stop()

model, ckpt, scaler = load_model()

# --- Show training metadata ---
col_a, col_b, col_c = st.columns(3)
col_a.metric("学習時 val_sharpe", f"{ckpt['val_sharpe']:+.4f}")
col_b.metric("Lookback (日)", ckpt["config"]["lookback"])
col_c.metric("予測ホライズン (日)", ckpt["config"]["horizon"])
st.caption(f"モデル: {ckpt['config']['arch']}  "
           f"hidden={ckpt['config']['hidden']}  "
           f"num_layers={ckpt['config']['num_layers']}  "
           f"学習日時={ckpt.get('trained_at', '?')}  "
           f"notes={ckpt.get('notes', '')}")

# --- Run prediction ---
if go_btn:
    with st.spinner(f"{ticker} のデータ取得中..."):
        df = fetch_history(ticker, end_date=datetime.combine(base_date, datetime.min.time()))
    if df is None or len(df) < LOOKBACK + 5:
        st.error(f"{ticker} のデータを十分に取得できませんでした。")
        st.stop()

    pred = predict_next_returns(model, scaler, df)
    if pred is None:
        st.error("特徴量の計算に必要なウィンドウ長 (60 営業日) に満たないデータでした。")
        st.stop()

    last_close = float(df["Close"].iloc[-1])
    last_date = df.index[-1]
    forecast = build_forecast_path(last_close, pred, last_date)

    cum_return = float(np.exp(pred.sum()) - 1.0)
    avg_daily = float(pred.mean())
    pred_vol = float(pred.std(ddof=1)) if len(pred) > 1 else 0.0

    # --- Metric cards ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最終終値", f"¥{last_close:,.1f}")
    m2.metric("20 営業日後の予測終値", f"¥{forecast['ForecastClose'].iloc[-1]:,.1f}",
              delta=f"{cum_return * 100:+.2f}%")
    m3.metric("予測日次リターン平均", f"{avg_daily * 100:+.3f}%")
    m4.metric("予測日次リターン σ", f"{pred_vol * 100:.3f}%")

    signal_text = ("**シグナル: LONG** (予測 20 日累積 log リターン > 0)"
                   if pred.sum() > 0 else
                   "**シグナル: FLAT** (予測 20 日累積 log リターン ≤ 0)")
    st.markdown(signal_text)

    # --- Chart ---
    hist = df.iloc[-60:][["Close"]].copy()
    hist.columns = ["Close"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines",
        name="過去 60 営業日",
        line=dict(color="#1f77b4", width=2),
    ))
    # connect the last historical point to the first forecast point
    connect_x = [hist.index[-1], forecast.index[0]]
    connect_y = [hist["Close"].iloc[-1], forecast["ForecastClose"].iloc[0]]
    fig.add_trace(go.Scatter(
        x=connect_x, y=connect_y, mode="lines",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast["ForecastClose"],
        mode="lines+markers",
        name="予測 20 営業日",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
        marker=dict(size=5),
    ))
    # ±1σ band: cumulative std scales as sqrt(day_idx)*pred_vol, rough confidence ribbon
    day_idx = np.arange(1, len(forecast) + 1)
    upper = forecast["ForecastClose"].values * np.exp(+pred_vol * np.sqrt(day_idx))
    lower = forecast["ForecastClose"].values * np.exp(-pred_vol * np.sqrt(day_idx))
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast.index, forecast.index[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor="rgba(255,127,14,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ (予測日次 σ)",
        hoverinfo="skip",
    ))
    fig.update_layout(
        title=f"{ticker}  過去 60 営業日 + 予測 20 営業日",
        xaxis_title="日付",
        yaxis_title="終値 (円)",
        hovermode="x unified",
        height=560,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Detail table ---
    with st.expander("予測の内訳 (20 営業日)"):
        detail = forecast.copy()
        detail["Daily log return (pred)"] = pred
        detail["Daily return % (pred)"] = (np.exp(pred) - 1) * 100
        detail["Cum return %"] = (np.exp(np.cumsum(pred)) - 1) * 100
        detail.index = detail.index.strftime("%Y-%m-%d")
        st.dataframe(detail.style.format({
            "ForecastClose": "¥{:,.1f}",
            "Daily log return (pred)": "{:+.5f}",
            "Daily return % (pred)": "{:+.3f}%",
            "Cum return %": "{:+.3f}%",
        }), use_container_width=True)

st.divider()
st.caption(
    "⚠️ このアプリの出力は学習データと 100 銘柄の東証プライム過去データから学んだ "
    "統計的な予測であり、将来のリターンを保証するものではありません。"
    "実際の投資判断にはご自身での十分な検討をお願いします。"
)
