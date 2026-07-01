import json
import logging
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from data import load_ohlcv
from indicators import macd, rsi, ichimoku
from universe import all_names

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Analyzer API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = "notification_config.json"
_ALL_NAMES: dict[str, str] = {}

def _get_name(ticker: str) -> str:
    global _ALL_NAMES
    if not _ALL_NAMES:
        _ALL_NAMES = all_names()
    return _ALL_NAMES.get(ticker, ticker)

def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def _save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


@app.get("/api/tickers")
def get_tickers():
    """Return all configured tickers with their display names and notification settings."""
    config = _load_config()
    tickers_cfg = config.get("tickers", {})
    results = []
    for ticker, t_cfg in tickers_cfg.items():
        results.append({
            "ticker": ticker,
            "name": _get_name(ticker),
            "notifications_enabled": t_cfg.get("notifications_enabled", True)
        })
    return results


@app.get("/api/chart/{ticker}")
def get_chart_data(ticker: str):
    """Return OHLCV data + indicators for a ticker."""
    df = load_ohlcv(ticker, interval="1d")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data not found for ticker")

    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    # Indicators
    macd_df = macd(df["Close"])
    rsi_series = rsi(df["Close"])
    df["macd"] = macd_df["macd"].fillna(0)
    df["macd_signal"] = macd_df["signal"].fillna(0)
    df["macd_hist"] = macd_df["hist"].fillna(0)
    df["rsi"] = rsi_series.fillna(50)

    # Ichimoku (needs High/Low/Close columns)
    try:
        ichi_df = ichimoku(df)
        for col in ["tenkan", "kijun", "span_a", "span_b", "chikou"]:
            if col in ichi_df.columns:
                df[col] = ichi_df[col].fillna(0)
    except Exception as e:
        logger.warning("Ichimoku calculation failed: %s", e)

    # Trendlines from config
    config = _load_config()
    ticker_cfg = config.get("tickers", {}).get(ticker, {})
    trendlines = ticker_cfg.get("trendlines", [])

    return {
        "ticker": ticker,
        "name": _get_name(ticker),
        "data": df.to_dict(orient="records"),
        "user_trendlines": trendlines,
    }


@app.post("/api/config/notifications")
def update_notifications(payload: Dict[str, Any]):
    """Enable or disable notifications for a specific ticker."""
    ticker = payload.get("ticker")
    enabled = payload.get("enabled", True)
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    config = _load_config()
    config.setdefault("tickers", {}).setdefault(ticker, {})
    config["tickers"][ticker]["notifications_enabled"] = enabled
    _save_config(config)
    return {"status": "success", "ticker": ticker, "notifications_enabled": enabled}


@app.post("/api/trendlines")
def save_trendlines(payload: Dict[str, Any]):
    """Save user-drawn trendlines for a specific ticker."""
    ticker = payload.get("ticker")
    trendlines = payload.get("trendlines", [])
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    config = _load_config()
    config.setdefault("tickers", {}).setdefault(ticker, {})
    config["tickers"][ticker]["trendlines"] = trendlines
    _save_config(config)
    return {"status": "success", "trendlines_count": len(trendlines)}
