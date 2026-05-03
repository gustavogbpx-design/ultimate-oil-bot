"""
WTI / USOIL Combined Bot v3
==========================

Purpose
-------
This is a combined version of your original 30-minute alert system plus a
separate weekend opening-spike probability module.

Important design change
-----------------------
Gemini is NOT the decision maker anymore.

The bot now calculates:
1. 30-minute intraday alert action using a transparent rules engine.
2. Weekend opening-spike probabilities using a separate transparent model.
3. Gemini is optional and only explains the already-calculated result.

Core outputs
------------
- Dashboard: http://localhost:8080
- Chart image: /chart
- 30-minute alert log: intraday_alert_log.csv
- Opening spike log: opening_spike_audit_log.csv
- Historical weekend cases: weekend_opening_history.csv

Environment variables
---------------------
Optional:
    GEMINI_API_KEY
    TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID
    PORT

Install:
    pip install -r requirements_wti_combined_bot_v3.txt

Run:
    python wti_combined_30min_opening_spike_bot_v3.py
"""

import os
import time
import csv
import math
import base64
import calendar
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import feedparser
import yfinance as yf
import mplfinance as mpf

from flask import Flask, render_template_string, send_file

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

TICKER = os.environ.get("WTI_TICKER", "CL=F")
PORT = int(os.environ.get("PORT", 8080))

# Main monitoring cadence
LOOP_SLEEP_SECONDS = int(os.environ.get("LOOP_SLEEP_SECONDS", 120))
MAIN_REPORT_INTERVAL_SECONDS = int(os.environ.get("MAIN_REPORT_INTERVAL_SECONDS", 1800))  # 30 minutes
FRESH_NEWS_SECONDS = int(os.environ.get("FRESH_NEWS_SECONDS", 900))  # 15 minutes
FAST_SPIKE_THRESHOLD = float(os.environ.get("FAST_SPIKE_THRESHOLD", 0.50))  # $0.50 move between checks

# Telegram behavior
MUTE_WAIT_SIGNALS = True
SEND_CHART_WITH_TELEGRAM = True

# If your broker price is different from Yahoo by more than this, be careful.
PRICE_DISAGREEMENT_LIMIT = 0.40

# Manual chart levels.
# Update these before major trading sessions.
MANUAL_LEVELS = {
    "major_support_zone": (94.33, 94.90),
    "bullish_reclaim_1": 95.57,
    "bullish_reclaim_zone_2": (96.40, 96.70),
    "resistance_zone_1": (98.53, 98.67),
    "resistance_2": 100.45,
    "lower_support_zone_1": (90.00, 91.50),
    "lower_support_2": 86.71,
}

# Opening-spike watch window.
# Many brokers reopen around Sunday evening US time, which can be Monday morning
# in Sri Lanka. This window is deliberately broad.
OPENING_SPIKE_ACTIVE_WEEKDAYS_UTC = {6, 0}  # Sunday=6, Monday=0
OPENING_SPIKE_START_HOUR_UTC = 20           # Sunday from 20:00 UTC
OPENING_SPIKE_END_HOUR_UTC = 3              # Monday until 03:59 UTC

# Files
CHART_FILE = "oil_chart.png"
INTRADAY_LOG_FILE = "intraday_alert_log.csv"
OPENING_SPIKE_LOG_FILE = "opening_spike_audit_log.csv"
WEEKEND_HISTORY_FILE = "weekend_opening_history.csv"


# =============================================================================
# 2. GLOBAL DASHBOARD STATE
# =============================================================================

app = Flask(__name__)

DASHBOARD_DATA: Dict[str, Any] = {
    "status": "Booting...",
    "last_update": "N/A",
    "price": 0.0,
    "trend": "WAITING",
    "rsi15": 0.0,
    "rsi1h": 0.0,
    "atr15": 0.0,
    "atr1h": 0.0,
    "ema_status_15m": "WAITING",
    "ema_status_1h": "WAITING",
    "news": [],
    "news_score": 0,
    "analysis": "Awaiting first model run...",
    "data_quality": {"status": "WAITING", "warnings": [], "penalty": 0},
    "intraday": {},
    "opening_spike": {},
    "factor_table": [],
    "opening_factor_table": [],
    "calibration": {},
}


# =============================================================================
# 3. DATA STRUCTURES
# =============================================================================

@dataclass
class Factor:
    name: str
    signal: str
    weight: float
    note: str


@dataclass
class EngineResult:
    action: str
    direction_bias: str
    score: float
    conviction: float
    risk_level: str
    entry: float
    stop: float
    target: float
    reason: str
    factors: List[Factor]


@dataclass
class OpeningSpikeResult:
    active: bool
    up_probability: float
    down_probability: float
    whipsaw_probability: float
    bias: str
    confidence: float
    reason: str
    factors: List[Factor]
    historical_summary: str


# =============================================================================
# 4. UTILITY FUNCTIONS
# =============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_str() -> str:
    return utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.iloc[-1]
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def format_money(x: float) -> str:
    return f"${x:.2f}"


def append_csv_row(file_path: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(file_path)
    fieldnames = list(row.keys())
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def clean_title(title: str) -> str:
    if not title:
        return ""
    # Google News RSS titles often look like "Title - Publisher"
    return title.split(" - ")[0].strip()


# =============================================================================
# 5. MARKET DATA
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 60:
        return df

    df = df.copy()
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < 60:
        return df

    close = df["Close"]
    df["RSI"] = RSIIndicator(close=close, window=14).rsi()

    macd = MACD(close=close)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    df["ATR"] = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    ).average_true_range()

    df["EMA21"] = EMAIndicator(close=close, window=21).ema_indicator()
    df["EMA50"] = EMAIndicator(close=close, window=50).ema_indicator()
    df["EMA200"] = EMAIndicator(close=close, window=200).ema_indicator() if len(df) >= 220 else np.nan

    df["ATR_MEAN_50"] = df["ATR"].rolling(50).mean()
    df["RET_1"] = df["Close"].diff()
    df["RET_4"] = df["Close"].diff(4)

    return df


def download_interval(period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(TICKER, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        df = flatten_yfinance_columns(df)
        df = ensure_utc_index(df)
        df = add_indicators(df)
        return df
    except Exception as e:
        print(f"Download error {interval}: {e}")
        return None


def get_market_bundle() -> Dict[str, Any]:
    """
    Returns both 15m and 1h data.
    The 30-minute alert engine uses 15m data but confirms with 1h.
    """
    df15 = download_interval("10d", "15m")
    df1h = download_interval("45d", "1h")

    price = 0.0
    if df15 is not None and not df15.empty:
        price = safe_float(df15["Close"].iloc[-1])
    elif df1h is not None and not df1h.empty:
        price = safe_float(df1h["Close"].iloc[-1])

    return {
        "df15": df15,
        "df1h": df1h,
        "price": price,
    }


# =============================================================================
# 6. DATA QUALITY CHECK
# =============================================================================

def check_data_quality(df15: Optional[pd.DataFrame], df1h: Optional[pd.DataFrame]) -> Dict[str, Any]:
    warnings: List[str] = []
    penalty = 0

    if df15 is None or df15.empty:
        warnings.append("15m data missing")
        penalty += 25
    else:
        if len(df15) < 80:
            warnings.append("15m data has limited history")
            penalty += 10
        last_ts = df15.index[-1].to_pydatetime()
        age_minutes = (utc_now() - last_ts).total_seconds() / 60
        if age_minutes > 90:
            warnings.append(f"15m data may be stale: {age_minutes:.0f} minutes old")
            penalty += 20

        atr15 = safe_float(df15["ATR"].iloc[-1]) if "ATR" in df15.columns else 0
        if atr15 <= 0:
            warnings.append("15m ATR unavailable")
            penalty += 15

    if df1h is None or df1h.empty:
        warnings.append("1h data missing")
        penalty += 20
    else:
        if len(df1h) < 80:
            warnings.append("1h data has limited history")
            penalty += 8

        atr1h = safe_float(df1h["ATR"].iloc[-1]) if "ATR" in df1h.columns else 0
        if atr1h <= 0:
            warnings.append("1h ATR unavailable")
            penalty += 10

    if df15 is not None and not df15.empty and df1h is not None and not df1h.empty:
        p15 = safe_float(df15["Close"].iloc[-1])
        p1h = safe_float(df1h["Close"].iloc[-1])
        if abs(p15 - p1h) > PRICE_DISAGREEMENT_LIMIT:
            warnings.append(f"15m/1h price disagreement: {abs(p15 - p1h):.2f}")
            penalty += 15

    penalty = int(clamp(penalty, 0, 80))
    status = "GOOD" if penalty <= 10 else "CAUTION" if penalty <= 30 else "POOR"

    return {
        "status": status,
        "warnings": warnings,
        "penalty": penalty,
    }


# =============================================================================
# 7. CHANNEL ENGINE
# =============================================================================

def calculate_universal_channel(slice_data: pd.DataFrame, cutoff_pct: float = 0.92):
    """
    Linear regression channel.
    Returns:
        exists, support_line, resistance_line, channel_position
    channel_position:
        0.0 = lower channel
        1.0 = upper channel
    """
    if slice_data is None or len(slice_data) < 20:
        return False, [], [], 0.5

    slice_data = slice_data.dropna(subset=["High", "Low", "Close"])
    if len(slice_data) < 20:
        return False, [], [], 0.5

    x = np.arange(len(slice_data))
    y = slice_data["Close"].values

    try:
        m, b = np.polyfit(x, y, 1)
    except Exception:
        return False, [], [], 0.5

    reg_line = m * x + b
    high_offsets = slice_data["High"].values - reg_line
    low_offsets = reg_line - slice_data["Low"].values

    sorted_highs = np.sort(high_offsets)
    sorted_lows = np.sort(low_offsets)

    if len(sorted_highs) > 0 and len(sorted_lows) > 0:
        cutoff_idx = int(len(sorted_highs) * cutoff_pct)
        cutoff_idx = min(cutoff_idx, len(sorted_highs) - 1)
        upper_offset = sorted_highs[cutoff_idx]
        lower_offset = sorted_lows[cutoff_idx]
    else:
        upper_offset, lower_offset = 1.0, 1.0

    upper_channel = reg_line + upper_offset
    lower_channel = reg_line - lower_offset

    date_start = slice_data.index[0]
    date_end = slice_data.index[-1]

    support_line = [(date_start, lower_channel[0]), (date_end, lower_channel[-1])]
    resistance_line = [(date_start, upper_channel[0]), (date_end, upper_channel[-1])]

    latest_price = safe_float(slice_data["Close"].iloc[-1])
    low_now = lower_channel[-1]
    high_now = upper_channel[-1]
    width = max(0.0001, high_now - low_now)
    channel_position = clamp((latest_price - low_now) / width, 0.0, 1.0)

    return True, support_line, resistance_line, channel_position


# =============================================================================
# 8. NEWS ENGINE
# =============================================================================

BULLISH_NEWS_KEYWORDS = {
    "tanker": 14,
    "hormuz": 25,
    "strait": 18,
    "attack": 16,
    "missile": 16,
    "drone": 12,
    "iran": 8,
    "israel": 8,
    "sanction": 14,
    "supply disruption": 22,
    "output cut": 18,
    "production cut": 18,
    "opec cut": 18,
    "inventory draw": 12,
    "drawdown": 12,
    "refinery outage": 12,
    "war": 18,
    "escalation": 18,
    "seize": 14,
    "shipping": 8,
    "red sea": 14,
}

BEARISH_NEWS_KEYWORDS = {
    "ceasefire": -24,
    "peace": -16,
    "talks": -10,
    "deal": -10,
    "diplomacy": -12,
    "de-escalation": -22,
    "output increase": -20,
    "production increase": -20,
    "opec increase": -18,
    "inventory build": -14,
    "demand weak": -16,
    "demand concerns": -14,
    "slowdown": -12,
    "recession": -14,
    "surplus": -18,
    "glut": -18,
}


def score_headline(title: str) -> Dict[str, Any]:
    text = title.lower()
    score = 0
    hits: List[str] = []

    for kw, weight in BULLISH_NEWS_KEYWORDS.items():
        if kw in text:
            score += weight
            hits.append(f"{kw}({weight:+})")

    for kw, weight in BEARISH_NEWS_KEYWORDS.items():
        if kw in text:
            score += weight
            hits.append(f"{kw}({weight:+})")

    score = int(clamp(score, -40, 40))
    direction = "BULLISH" if score > 5 else "BEARISH" if score < -5 else "NEUTRAL"
    return {"score": score, "direction": direction, "hits": hits}


def get_news() -> Tuple[List[Dict[str, Any]], List[Any], int]:
    """
    Returns:
        scored headlines, raw feed entries, aggregate news score
    """
    try:
        query = '("Crude Oil" OR "WTI" OR "OPEC" OR "Middle East" OR "Iran" OR "Hormuz") when:2d'
        base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
        final_url = base_url.format(requests.utils.quote(query))
        feed = feedparser.parse(final_url)

        if not feed.entries:
            return [], [], 0

        current_time = calendar.timegm(time.gmtime())
        headlines: List[Dict[str, Any]] = []
        raw_entries: List[Any] = []

        for entry in feed.entries:
            if "published_parsed" not in entry:
                continue

            entry_time = calendar.timegm(entry.published_parsed)
            age_hours = (current_time - entry_time) / 3600

            if age_hours <= 48:
                title = clean_title(entry.get("title", ""))
                scoring = score_headline(title)

                # Newer headlines matter more.
                freshness_multiplier = 1.0 if age_hours <= 6 else 0.7 if age_hours <= 24 else 0.4
                weighted_score = int(scoring["score"] * freshness_multiplier)

                headlines.append({
                    "title": title,
                    "time": entry.get("published", ""),
                    "age_hours": round(age_hours, 1),
                    "link": entry.get("link", ""),
                    "score": weighted_score,
                    "direction": scoring["direction"],
                    "hits": ", ".join(scoring["hits"]) if scoring["hits"] else "none",
                })
                raw_entries.append(entry)

            if len(headlines) >= 10:
                break

        aggregate = int(clamp(sum(h["score"] for h in headlines), -60, 60))
        return headlines, raw_entries, aggregate

    except Exception as e:
        print(f"News fetch error: {e}")
        return [], [], 0


# =============================================================================
# 9. CHART CREATION
# =============================================================================

def create_chart(df1h: Optional[pd.DataFrame], df15: Optional[pd.DataFrame]) -> Optional[str]:
    """
    Chart uses 1H data when available because it is cleaner for structure.
    Falls back to 15m.
    """
    plot_data = df1h.tail(250) if df1h is not None and not df1h.empty else None
    if plot_data is None and df15 is not None and not df15.empty:
        plot_data = df15.tail(250)

    if plot_data is None or plot_data.empty:
        return None

    try:
        recent_low = safe_float(plot_data.tail(250)["Low"].min())
        recent_high = safe_float(plot_data.tail(250)["High"].max())

        horizontal_lines = [
            recent_low,
            recent_high,
            MANUAL_LEVELS["major_support_zone"][0],
            MANUAL_LEVELS["major_support_zone"][1],
            MANUAL_LEVELS["bullish_reclaim_1"],
            MANUAL_LEVELS["resistance_zone_1"][0],
            MANUAL_LEVELS["resistance_zone_1"][1],
        ]

        macro_exists, macro_sup, macro_res, _ = calculate_universal_channel(plot_data, cutoff_pct=0.92)
        micro_source = plot_data.tail(70)
        micro_exists, micro_sup, micro_res, _ = calculate_universal_channel(micro_source, cutoff_pct=0.80)

        mc = mpf.make_marketcolors(
            up="#00E676",
            down="#D500F9",
            edge="inherit",
            wick="inherit",
            volume="in",
        )
        iq_style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor="#0f172a",
            edgecolor="#1e293b",
            figcolor="#0f172a",
        )

        kwargs = dict(
            type="candle",
            style=iq_style,
            volume=False,
            mav=(21, 50),
            hlines=dict(
                hlines=horizontal_lines,
                colors=["#ffcc00"] * len(horizontal_lines),
                linestyle="--",
                linewidths=0.8,
            ),
            savefig=dict(fname=CHART_FILE, dpi=120, bbox_inches="tight"),
        )

        lines_to_draw: List[Any] = []
        line_colors: List[str] = []
        line_styles: List[str] = []

        if macro_exists:
            lines_to_draw.extend([macro_sup, macro_res])
            line_colors.extend(["white", "gray"])
            line_styles.extend(["-", "-"])

        if micro_exists:
            lines_to_draw.extend([micro_sup, micro_res])
            line_colors.extend(["#00aaff", "#00aaff"])
            line_styles.extend(["--", "--"])

        if lines_to_draw:
            kwargs["alines"] = dict(
                alines=lines_to_draw,
                colors=line_colors,
                linestyle=line_styles,
                linewidths=1.5,
            )

        mpf.plot(plot_data, **kwargs)
        return CHART_FILE

    except Exception as e:
        print(f"Chart error: {e}")
        return None


# =============================================================================
# 10. 30-MINUTE INTRADAY ALERT ENGINE
# =============================================================================

def proximity_score(price: float, low: float, high: float, atr: float) -> float:
    """
    Positive if price is near or inside the zone.
    """
    if atr <= 0:
        atr = 0.5
    if low <= price <= high:
        return 1.0
    distance = min(abs(price - low), abs(price - high))
    return clamp(1 - distance / (1.5 * atr), 0, 1)


def calculate_intraday_alert(
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    news_score: int,
    data_quality: Dict[str, Any],
    last_price: float,
) -> EngineResult:

    factors: List[Factor] = []
    price = safe_float(df15["Close"].iloc[-1]) if df15 is not None and not df15.empty else 0.0

    if price <= 0:
        return EngineResult(
            action="STRICT WAIT",
            direction_bias="NONE",
            score=0,
            conviction=0,
            risk_level="HIGH",
            entry=0,
            stop=0,
            target=0,
            reason="No reliable price data.",
            factors=[Factor("Data", "Blocked", -50, "No price data")],
        )

    latest15 = df15.iloc[-1]
    latest1h = df1h.iloc[-1] if df1h is not None and not df1h.empty else latest15

    rsi15 = safe_float(latest15.get("RSI"))
    rsi1h = safe_float(latest1h.get("RSI"))
    atr15 = safe_float(latest15.get("ATR"), 0.35)
    atr1h = safe_float(latest1h.get("ATR"), atr15)
    ema21_15 = safe_float(latest15.get("EMA21"))
    ema50_15 = safe_float(latest15.get("EMA50"))
    ema21_1h = safe_float(latest1h.get("EMA21"))
    ema50_1h = safe_float(latest1h.get("EMA50"))
    macd_hist15 = safe_float(latest15.get("MACD_HIST"))
    macd_hist1h = safe_float(latest1h.get("MACD_HIST"))
    atr_mean50 = safe_float(latest15.get("ATR_MEAN_50"), atr15)

    score = 0.0

    # 1H trend confirmation
    if ema21_1h > ema50_1h:
        score += 12
        factors.append(Factor("1H EMA trend", "Bullish", 12, "EMA21 is above EMA50 on 1H"))
    elif ema21_1h < ema50_1h:
        score -= 12
        factors.append(Factor("1H EMA trend", "Bearish", -12, "EMA21 is below EMA50 on 1H"))
    else:
        factors.append(Factor("1H EMA trend", "Neutral", 0, "No clear 1H EMA separation"))

    # 15m trend
    if ema21_15 > ema50_15:
        score += 9
        factors.append(Factor("15m EMA trend", "Bullish", 9, "Short-term EMA is positive"))
    elif ema21_15 < ema50_15:
        score -= 9
        factors.append(Factor("15m EMA trend", "Bearish", -9, "Short-term EMA is negative"))
    else:
        factors.append(Factor("15m EMA trend", "Neutral", 0, "No clear short-term EMA separation"))

    # RSI momentum
    if 52 <= rsi15 <= 68:
        score += 7
        factors.append(Factor("15m RSI", "Bullish", 7, f"RSI {rsi15:.1f} supports upside momentum"))
    elif 32 <= rsi15 <= 48:
        score -= 7
        factors.append(Factor("15m RSI", "Bearish", -7, f"RSI {rsi15:.1f} supports downside momentum"))
    elif rsi15 > 72:
        score -= 5
        factors.append(Factor("15m RSI", "Overbought risk", -5, f"RSI {rsi15:.1f} increases pullback risk"))
    elif rsi15 < 28:
        score += 5
        factors.append(Factor("15m RSI", "Oversold bounce risk", 5, f"RSI {rsi15:.1f} increases bounce risk"))
    else:
        factors.append(Factor("15m RSI", "Neutral", 0, f"RSI {rsi15:.1f} is not decisive"))

    # MACD
    if macd_hist15 > 0 and macd_hist1h > 0:
        score += 8
        factors.append(Factor("MACD histogram", "Bullish", 8, "15m and 1H MACD histograms are positive"))
    elif macd_hist15 < 0 and macd_hist1h < 0:
        score -= 8
        factors.append(Factor("MACD histogram", "Bearish", -8, "15m and 1H MACD histograms are negative"))
    elif macd_hist15 > 0:
        score += 3
        factors.append(Factor("MACD histogram", "Mild bullish", 3, "15m MACD is positive but 1H is not aligned"))
    elif macd_hist15 < 0:
        score -= 3
        factors.append(Factor("MACD histogram", "Mild bearish", -3, "15m MACD is negative but 1H is not aligned"))
    else:
        factors.append(Factor("MACD histogram", "Neutral", 0, "MACD is flat"))

    # Manual levels
    support_low, support_high = MANUAL_LEVELS["major_support_zone"]
    support_prox = proximity_score(price, support_low, support_high, atr1h)
    if support_prox > 0.65:
        w = 12 * support_prox
        score += w
        factors.append(Factor("Manual support zone", "Bullish", round(w, 1), f"Price is near/supporting {support_low:.2f}-{support_high:.2f}"))
    else:
        factors.append(Factor("Manual support zone", "Neutral", 0, "Price is not close enough to major support"))

    reclaim1 = MANUAL_LEVELS["bullish_reclaim_1"]
    if price > reclaim1:
        score += 10
        factors.append(Factor("Reclaim level 1", "Bullish", 10, f"Price is above {reclaim1:.2f}"))
    else:
        score -= 5
        factors.append(Factor("Reclaim level 1", "Mild bearish", -5, f"Price is still below {reclaim1:.2f}"))

    reclaim2_low, reclaim2_high = MANUAL_LEVELS["bullish_reclaim_zone_2"]
    if price > reclaim2_high:
        score += 10
        factors.append(Factor("Reclaim zone 2", "Bullish", 10, f"Price is above {reclaim2_low:.2f}-{reclaim2_high:.2f}"))
    elif reclaim2_low <= price <= reclaim2_high:
        score += 4
        factors.append(Factor("Reclaim zone 2", "Testing", 4, "Price is testing upper reclaim zone"))
    else:
        factors.append(Factor("Reclaim zone 2", "Not reclaimed", 0, "Upper reclaim zone not reached"))

    # Resistance proximity
    res_low, res_high = MANUAL_LEVELS["resistance_zone_1"]
    res_prox = proximity_score(price, res_low, res_high, atr1h)
    if res_prox > 0.65:
        w = -10 * res_prox
        score += w
        factors.append(Factor("Resistance zone", "Upside risk", round(w, 1), f"Price is near {res_low:.2f}-{res_high:.2f} resistance"))

    # Regression channel position
    macro_exists, _, _, channel_pos_1h = calculate_universal_channel(df1h.tail(250), 0.92)
    micro_exists, _, _, channel_pos_15 = calculate_universal_channel(df15.tail(160), 0.80)

    if macro_exists:
        if channel_pos_1h < 0.25:
            score += 8
            factors.append(Factor("1H channel position", "Bullish support area", 8, "Price is near lower macro channel"))
        elif channel_pos_1h > 0.78:
            score -= 8
            factors.append(Factor("1H channel position", "Resistance area", -8, "Price is near upper macro channel"))
        else:
            factors.append(Factor("1H channel position", "Middle", 0, f"Channel position {channel_pos_1h:.2f}"))

    if micro_exists:
        if channel_pos_15 < 0.22:
            score += 5
            factors.append(Factor("15m channel position", "Short-term bounce zone", 5, "Price is near lower micro channel"))
        elif channel_pos_15 > 0.82:
            score -= 5
            factors.append(Factor("15m channel position", "Short-term rejection zone", -5, "Price is near upper micro channel"))
        else:
            factors.append(Factor("15m channel position", "Middle", 0, f"Micro channel position {channel_pos_15:.2f}"))

    # Volatility expansion
    if atr_mean50 > 0:
        atr_ratio = atr15 / atr_mean50
        if atr_ratio > 1.35:
            # More volatility means signal is more dangerous, not automatically buy/sell.
            if score > 0:
                score += 3
            elif score < 0:
                score -= 3
            factors.append(Factor("ATR expansion", "High volatility", 3 if score > 0 else -3 if score < 0 else 0, f"ATR ratio {atr_ratio:.2f}; stronger but riskier signal"))
        elif atr_ratio < 0.70:
            score *= 0.85
            factors.append(Factor("ATR compression", "Low energy", -3, f"ATR ratio {atr_ratio:.2f}; breakout energy is weak"))
        else:
            factors.append(Factor("ATR state", "Normal", 0, f"ATR ratio {atr_ratio:.2f}"))

    # Fast spike between bot checks
    if last_price and last_price > 0:
        diff = price - last_price
        if abs(diff) >= FAST_SPIKE_THRESHOLD:
            if diff > 0:
                score += 12
                factors.append(Factor("Fast spike detector", "Bullish impulse", 12, f"Price moved +{diff:.2f} since last check"))
            else:
                score -= 12
                factors.append(Factor("Fast spike detector", "Bearish impulse", -12, f"Price moved {diff:.2f} since last check"))

    # News pressure is only one input in 30m mode.
    news_weight = clamp(news_score * 0.25, -15, 15)
    score += news_weight
    if news_weight > 4:
        factors.append(Factor("News pressure", "Bullish", round(news_weight, 1), f"Aggregate news score {news_score:+}"))
    elif news_weight < -4:
        factors.append(Factor("News pressure", "Bearish", round(news_weight, 1), f"Aggregate news score {news_score:+}"))
    else:
        factors.append(Factor("News pressure", "Neutral", round(news_weight, 1), f"Aggregate news score {news_score:+}"))

    # Data-quality penalty reduces conviction, not directly direction.
    data_penalty = data_quality.get("penalty", 0)
    if data_penalty > 0:
        factors.append(Factor("Data quality", data_quality.get("status", "UNKNOWN"), -data_penalty, "; ".join(data_quality.get("warnings", [])) or "Minor data penalty"))

    raw_score = score
    score = clamp(score, -80, 80)

    # Determine action
    abs_score = abs(score)
    blocked = data_quality.get("status") == "POOR" or data_penalty >= 45

    if blocked:
        action = "STRICT WAIT"
        direction_bias = "BLOCKED"
        reason = "Signal blocked by poor data quality."
    elif score >= 28:
        action = "BUY"
        direction_bias = "UP"
        reason = "Bullish intraday score is strong enough for a 30-minute alert."
    elif score <= -28:
        action = "SELL"
        direction_bias = "DOWN"
        reason = "Bearish intraday score is strong enough for a 30-minute alert."
    else:
        action = "STRICT WAIT"
        direction_bias = "MIXED"
        reason = "Score is not strong enough; wait for cleaner confirmation."

    # Conviction is absolute score with data penalty.
    conviction = clamp(35 + abs_score - data_penalty * 0.7, 0, 92)
    if action == "STRICT WAIT" and not blocked:
        conviction = clamp(45 - abs_score * 0.4 - data_penalty * 0.5, 10, 55)
    if blocked:
        conviction = clamp(20 - data_penalty * 0.2, 0, 25)

    # Risk level
    atr_pct = (atr15 / price) * 100 if price > 0 else 0
    if data_penalty >= 35 or atr_pct > 1.0 or abs(news_score) > 50:
        risk_level = "HIGH"
    elif data_penalty >= 15 or atr_pct > 0.55 or abs(news_score) > 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Stops and targets
    min_atr = max(atr15, 0.25)
    if action == "BUY":
        stop = price - 1.30 * min_atr
        target = price + 2.10 * min_atr
    elif action == "SELL":
        stop = price + 1.30 * min_atr
        target = price - 2.10 * min_atr
    else:
        # Show hypothetical levels, not a trade.
        if score >= 0:
            stop = price - 1.30 * min_atr
            target = price + 2.10 * min_atr
        else:
            stop = price + 1.30 * min_atr
            target = price - 2.10 * min_atr

    return EngineResult(
        action=action,
        direction_bias=direction_bias,
        score=round(score, 1),
        conviction=round(conviction, 1),
        risk_level=risk_level,
        entry=round(price, 2),
        stop=round(stop, 2),
        target=round(target, 2),
        reason=reason,
        factors=factors,
    )


# =============================================================================
# 11. WEEKEND OPENING-SPIKE MODULE
# =============================================================================

def is_opening_spike_watch_window() -> bool:
    now = utc_now()
    wd = now.weekday()
    hr = now.hour

    # Sunday after configured hour
    if wd == 6 and hr >= OPENING_SPIKE_START_HOUR_UTC:
        return True
    # Monday before configured end hour
    if wd == 0 and hr <= OPENING_SPIKE_END_HOUR_UTC:
        return True
    return False


def build_weekend_history_from_15m(df15: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Extract weekend/gap reopen cases from 15m data.
    This is not a perfect exchange-level dataset, but it gives the bot a local
    evidence layer instead of relying only on AI.

    Case logic:
    - Find gaps between consecutive bars greater than 24 hours.
    - Previous bar close = pre-weekend close.
    - New bar open = reopen price.
    - First two 15m bars are used to classify first 30m spike.
    """
    columns = [
        "reopen_time_utc",
        "pre_close",
        "reopen_open",
        "gap",
        "first_30m_high",
        "first_30m_low",
        "first_30m_close",
        "spike_size_up",
        "spike_size_down",
        "spike_direction",
    ]

    if df15 is None or df15.empty or len(df15) < 100:
        return pd.DataFrame(columns=columns)

    records: List[Dict[str, Any]] = []
    idx = df15.index

    for i in range(1, len(idx) - 3):
        gap_hours = (idx[i] - idx[i - 1]).total_seconds() / 3600
        if gap_hours >= 24:
            pre = df15.iloc[i - 1]
            first = df15.iloc[i]
            next1 = df15.iloc[i + 1:i + 3]

            pre_close = safe_float(pre["Close"])
            reopen_open = safe_float(first["Open"])
            first_high = safe_float(pd.concat([first.to_frame().T, next1])["High"].max())
            first_low = safe_float(pd.concat([first.to_frame().T, next1])["Low"].min())
            first_close = safe_float(df15.iloc[i + 2]["Close"]) if i + 2 < len(df15) else safe_float(first["Close"])

            spike_up = first_high - reopen_open
            spike_down = reopen_open - first_low
            threshold = 0.20

            if spike_up >= spike_down + threshold:
                direction = "UP"
            elif spike_down >= spike_up + threshold:
                direction = "DOWN"
            else:
                direction = "WHIPSAW"

            records.append({
                "reopen_time_utc": idx[i].strftime("%Y-%m-%d %H:%M:%S UTC"),
                "pre_close": round(pre_close, 2),
                "reopen_open": round(reopen_open, 2),
                "gap": round(reopen_open - pre_close, 2),
                "first_30m_high": round(first_high, 2),
                "first_30m_low": round(first_low, 2),
                "first_30m_close": round(first_close, 2),
                "spike_size_up": round(spike_up, 2),
                "spike_size_down": round(spike_down, 2),
                "spike_direction": direction,
            })

    hist = pd.DataFrame(records, columns=columns)

    # Save for audit/reuse.
    if not hist.empty:
        try:
            existing = pd.read_csv(WEEKEND_HISTORY_FILE) if os.path.exists(WEEKEND_HISTORY_FILE) else pd.DataFrame(columns=columns)
            combined = pd.concat([existing, hist], ignore_index=True)
            combined = combined.drop_duplicates(subset=["reopen_time_utc"], keep="last")
            combined.to_csv(WEEKEND_HISTORY_FILE, index=False)
        except Exception as e:
            print(f"Weekend history save error: {e}")

    return hist


def load_weekend_history() -> pd.DataFrame:
    if os.path.exists(WEEKEND_HISTORY_FILE):
        try:
            return pd.read_csv(WEEKEND_HISTORY_FILE)
        except Exception:
            pass
    return pd.DataFrame()


def summarize_weekend_history(hist: pd.DataFrame) -> Tuple[str, float]:
    if hist is None or hist.empty or "spike_direction" not in hist.columns:
        return "No usable weekend history yet.", 0

    recent = hist.tail(20)
    total = len(recent)
    up = int((recent["spike_direction"] == "UP").sum())
    down = int((recent["spike_direction"] == "DOWN").sum())
    whipsaw = int((recent["spike_direction"] == "WHIPSAW").sum())

    up_rate = up / total if total else 0
    summary = f"Recent weekend cases: {total}; UP={up}, DOWN={down}, WHIPSAW={whipsaw}."
    return summary, up_rate


def calculate_opening_spike_probability(
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    news_score: int,
    data_quality: Dict[str, Any],
) -> OpeningSpikeResult:

    active = is_opening_spike_watch_window()
    factors: List[Factor] = []

    price = safe_float(df15["Close"].iloc[-1]) if df15 is not None and not df15.empty else 0.0
    latest1h = df1h.iloc[-1] if df1h is not None and not df1h.empty else None

    # Base probabilities before evidence.
    up = 37.0
    down = 37.0
    whipsaw = 26.0

    if price <= 0 or latest1h is None:
        return OpeningSpikeResult(
            active=active,
            up_probability=33.0,
            down_probability=33.0,
            whipsaw_probability=34.0,
            bias="NO DATA",
            confidence=0.0,
            reason="Opening-spike module has insufficient data.",
            factors=[Factor("Data", "Blocked", -50, "Missing price/1H data")],
            historical_summary="No usable weekend history yet.",
        )

    rsi1h = safe_float(latest1h.get("RSI"))
    atr1h = safe_float(latest1h.get("ATR"), 0.6)
    ema21_1h = safe_float(latest1h.get("EMA21"))
    ema50_1h = safe_float(latest1h.get("EMA50"))

    # Technical support/reclaim levels
    support_low, support_high = MANUAL_LEVELS["major_support_zone"]
    support_prox = proximity_score(price, support_low, support_high, atr1h)

    if support_prox > 0.65:
        adj = 10 * support_prox
        up += adj
        down -= adj * 0.5
        factors.append(Factor("Opening support location", "UP bias", round(adj, 1), f"Price is near {support_low:.2f}-{support_high:.2f} support"))
    else:
        factors.append(Factor("Opening support location", "Neutral", 0, "Price is not sitting directly on the manual support zone"))

    reclaim1 = MANUAL_LEVELS["bullish_reclaim_1"]
    if price > reclaim1:
        up += 7
        down -= 3
        factors.append(Factor("Opening reclaim level", "UP bias", 7, f"Price is above {reclaim1:.2f}"))
    else:
        up -= 5
        down += 4
        factors.append(Factor("Opening reclaim level", "DOWN risk", -5, f"Price is below {reclaim1:.2f}"))

    if ema21_1h > ema50_1h:
        up += 7
        down -= 4
        factors.append(Factor("1H EMA context", "UP bias", 7, "EMA21 above EMA50"))
    elif ema21_1h < ema50_1h:
        up -= 7
        down += 7
        factors.append(Factor("1H EMA context", "DOWN bias", -7, "EMA21 below EMA50"))
    else:
        factors.append(Factor("1H EMA context", "Neutral", 0, "EMA trend is flat"))

    if rsi1h > 58:
        up += 5
        down -= 2
        factors.append(Factor("1H RSI", "UP momentum", 5, f"RSI {rsi1h:.1f}"))
    elif rsi1h < 42:
        up -= 5
        down += 5
        factors.append(Factor("1H RSI", "DOWN momentum", -5, f"RSI {rsi1h:.1f}"))
    else:
        whipsaw += 2
        factors.append(Factor("1H RSI", "Neutral", 0, f"RSI {rsi1h:.1f}"))

    # News is more important in opening-spike mode than intraday mode.
    news_adj = clamp(news_score * 0.45, -25, 25)
    if news_adj > 5:
        up += news_adj
        down -= news_adj * 0.3
        factors.append(Factor("Weekend news pressure", "UP catalyst", round(news_adj, 1), f"Aggregate news score {news_score:+}"))
    elif news_adj < -5:
        down += abs(news_adj)
        up -= abs(news_adj) * 0.4
        factors.append(Factor("Weekend news pressure", "DOWN catalyst", round(news_adj, 1), f"Aggregate news score {news_score:+}"))
    else:
        whipsaw += 3
        factors.append(Factor("Weekend news pressure", "Neutral", round(news_adj, 1), f"Aggregate news score {news_score:+}"))

    # Historical weekend cases
    new_hist = build_weekend_history_from_15m(df15)
    stored_hist = load_weekend_history()
    hist = stored_hist if not stored_hist.empty else new_hist
    hist_summary, up_rate = summarize_weekend_history(hist)

    if not hist.empty:
        # Convert recent UP rate into modest adjustment.
        hist_adj = (up_rate - 0.50) * 18
        up += hist_adj
        down -= hist_adj
        factors.append(Factor("Recent weekend behavior", "Historical adjustment", round(hist_adj, 1), hist_summary))
    else:
        factors.append(Factor("Recent weekend behavior", "Not available", 0, hist_summary))

    # Data quality affects confidence and whipsaw.
    dq_penalty = data_quality.get("penalty", 0)
    if dq_penalty > 25:
        whipsaw += 8
        up -= 3
        down -= 3
        factors.append(Factor("Data quality", data_quality.get("status", "UNKNOWN"), -dq_penalty, "; ".join(data_quality.get("warnings", []))))

    # Normalize
    up = clamp(up, 5, 90)
    down = clamp(down, 5, 90)
    whipsaw = clamp(whipsaw, 5, 60)

    total = up + down + whipsaw
    up_p = up / total * 100
    down_p = down / total * 100
    whip_p = whipsaw / total * 100

    # Bias
    if up_p >= down_p + 10 and up_p >= whip_p + 8:
        bias = "SPIKE UP"
    elif down_p >= up_p + 10 and down_p >= whip_p + 8:
        bias = "SPIKE DOWN"
    else:
        bias = "NO CLEAN EDGE"

    edge = max(up_p, down_p) - min(up_p, down_p)
    confidence = clamp(40 + edge * 0.8 - dq_penalty * 0.5, 0, 88)

    if bias == "SPIKE UP":
        reason = "Opening-spike model leans UP from combined technical location, news pressure, and historical adjustment."
    elif bias == "SPIKE DOWN":
        reason = "Opening-spike model leans DOWN from combined technical location, news pressure, and historical adjustment."
    else:
        reason = "Opening-spike model does not show enough edge; whipsaw/no-clean-open risk is high."

    return OpeningSpikeResult(
        active=active,
        up_probability=round(up_p, 1),
        down_probability=round(down_p, 1),
        whipsaw_probability=round(whip_p, 1),
        bias=bias,
        confidence=round(confidence, 1),
        reason=reason,
        factors=factors,
        historical_summary=hist_summary,
    )


# =============================================================================
# 12. GEMINI EXPLANATION LAYER
# =============================================================================

def get_valid_gemini_model() -> str:
    if not GEMINI_KEY:
        return "models/gemini-1.5-flash"

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "models" in data:
            for model in data["models"]:
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    if "flash" in model["name"]:
                        return model["name"]
            return data["models"][0]["name"]
    except Exception:
        pass
    return "models/gemini-1.5-flash"


def fallback_explanation(intraday: EngineResult, opening: OpeningSpikeResult, news_score: int) -> str:
    return (
        f"🎯 30-MIN ALERT: {intraday.action} | Conviction {intraday.conviction:.1f}% | Risk {intraday.risk_level}<br>"
        f"Reason: {intraday.reason}<br><br>"
        f"🕯️ OPENING SPIKE MODULE: {opening.bias} | UP {opening.up_probability:.1f}% / "
        f"DOWN {opening.down_probability:.1f}% / WHIPSAW {opening.whipsaw_probability:.1f}% | "
        f"Confidence {opening.confidence:.1f}%<br>"
        f"Reason: {opening.reason}<br><br>"
        f"📰 News pressure score: {news_score:+}"
    )


def ai_explain_result(
    intraday: EngineResult,
    opening: OpeningSpikeResult,
    headlines: List[Dict[str, Any]],
    data_quality: Dict[str, Any],
    chart_file: Optional[str],
) -> str:
    """
    Gemini explains the already-calculated model result.
    It is explicitly forbidden from changing action/probabilities.
    """
    if not GEMINI_KEY:
        return fallback_explanation(intraday, opening, sum(h.get("score", 0) for h in headlines))

    model_name = get_valid_gemini_model()
    news_text = "\n".join([f"- {h['title']} | score {h['score']:+} | {h['direction']}" for h in headlines[:8]])

    intraday_factors = "\n".join([f"- {f.name}: {f.signal}, weight {f.weight:+}, {f.note}" for f in intraday.factors[:12]])
    opening_factors = "\n".join([f"- {f.name}: {f.signal}, weight {f.weight:+}, {f.note}" for f in opening.factors[:12]])

    prompt = f"""
You are an explanation layer for a trading-alert dashboard.

CRITICAL RULE:
Do NOT change the model action, probabilities, stops, targets, or confidence.
Only explain the already-calculated result.

CURRENT TIME: {utc_now_str()}

30-MINUTE ALERT MODEL:
Action: {intraday.action}
Direction bias: {intraday.direction_bias}
Score: {intraday.score}
Conviction: {intraday.conviction}%
Risk: {intraday.risk_level}
Entry: {intraday.entry}
Stop: {intraday.stop}
Target: {intraday.target}
Reason: {intraday.reason}

30-MIN FACTORS:
{intraday_factors}

OPENING SPIKE MODULE:
Active watch window: {opening.active}
Bias: {opening.bias}
UP probability: {opening.up_probability}%
DOWN probability: {opening.down_probability}%
WHIPSAW probability: {opening.whipsaw_probability}%
Confidence: {opening.confidence}%
Reason: {opening.reason}
Historical summary: {opening.historical_summary}

OPENING SPIKE FACTORS:
{opening_factors}

DATA QUALITY:
Status: {data_quality.get('status')}
Warnings: {data_quality.get('warnings')}
Penalty: {data_quality.get('penalty')}

NEWS:
{news_text}

OUTPUT FORMAT:
🎯 30-MIN ALERT
- Action:
- Conviction:
- Why:
- Risk:

🕯️ OPENING SPIKE MODULE
- Bias:
- Probability:
- Why:
- Use this only for weekend reopen, not normal intraday trading.

✅ DEFENSE SUMMARY
Maximum 3 bullets explaining why this is more defendable than AI-only decision making.
"""

    parts: List[Dict[str, Any]] = [{"text": prompt}]

    if chart_file and os.path.exists(chart_file):
        try:
            with open(chart_file, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")
            parts.append({"inline_data": {"mime_type": "image/png", "data": encoded_image}})
        except Exception:
            pass

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_KEY}"
        resp = requests.post(
            url,
            json={"contents": [{"parts": parts}]},
            headers={"Content-Type": "application/json"},
            timeout=25,
        )
        if resp.status_code == 200:
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return text.replace("\n", "<br>")
        return fallback_explanation(intraday, opening, sum(h.get("score", 0) for h in headlines))
    except Exception:
        return fallback_explanation(intraday, opening, sum(h.get("score", 0) for h in headlines))


# =============================================================================
# 13. TELEGRAM
# =============================================================================

def telegram_enabled() -> bool:
    return bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram_message(text: str, chart_file: Optional[str] = None) -> None:
    if not telegram_enabled():
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

    try:
        if SEND_CHART_WITH_TELEGRAM and chart_file and os.path.exists(chart_file):
            with open(chart_file, "rb") as f:
                requests.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": TELEGRAM_CHAT_ID},
                    files={"photo": f},
                    timeout=15,
                )

        # Telegram markdown can break easily with symbols, so plain text is safer.
        requests.post(
            f"{base_url}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=15,
        )
    except Exception as e:
        print(f"Telegram error: {e}")


def build_telegram_text(
    alert_reason: str,
    intraday: EngineResult,
    opening: OpeningSpikeResult,
    data_quality: Dict[str, Any],
    news_score: int,
) -> str:
    return (
        f"WTI / USOIL ALERT: {alert_reason}\n"
        f"Time: {utc_now_str()}\n\n"
        f"30-MIN SYSTEM\n"
        f"Action: {intraday.action}\n"
        f"Bias: {intraday.direction_bias}\n"
        f"Score: {intraday.score}\n"
        f"Conviction: {intraday.conviction}%\n"
        f"Risk: {intraday.risk_level}\n"
        f"Entry: {intraday.entry}\n"
        f"Stop: {intraday.stop}\n"
        f"Target: {intraday.target}\n"
        f"Reason: {intraday.reason}\n\n"
        f"OPENING SPIKE MODULE\n"
        f"Active: {opening.active}\n"
        f"Bias: {opening.bias}\n"
        f"UP: {opening.up_probability}% | DOWN: {opening.down_probability}% | WHIPSAW: {opening.whipsaw_probability}%\n"
        f"Confidence: {opening.confidence}%\n"
        f"Reason: {opening.reason}\n\n"
        f"News score: {news_score:+}\n"
        f"Data quality: {data_quality.get('status')} | penalty {data_quality.get('penalty')}\n"
    )


# =============================================================================
# 14. LOGGING AND CALIBRATION
# =============================================================================

def log_intraday_result(intraday: EngineResult, news_score: int, data_quality: Dict[str, Any]) -> None:
    row = {
        "timestamp_utc": utc_now_str(),
        "action": intraday.action,
        "direction_bias": intraday.direction_bias,
        "score": intraday.score,
        "conviction": intraday.conviction,
        "risk_level": intraday.risk_level,
        "entry": intraday.entry,
        "stop": intraday.stop,
        "target": intraday.target,
        "news_score": news_score,
        "data_quality_status": data_quality.get("status"),
        "data_quality_penalty": data_quality.get("penalty"),
        "actual_result_manual_fill": "",
    }
    append_csv_row(INTRADAY_LOG_FILE, row)


def log_opening_spike_result(opening: OpeningSpikeResult, price: float, news_score: int) -> None:
    row = {
        "timestamp_utc": utc_now_str(),
        "active": opening.active,
        "bias": opening.bias,
        "up_probability": opening.up_probability,
        "down_probability": opening.down_probability,
        "whipsaw_probability": opening.whipsaw_probability,
        "confidence": opening.confidence,
        "price": price,
        "news_score": news_score,
        "actual_opening_spike_manual_fill": "",
    }
    append_csv_row(OPENING_SPIKE_LOG_FILE, row)


def get_calibration_summary() -> Dict[str, Any]:
    """
    Reads logs if user manually fills actual_result_manual_fill.
    This creates future defense evidence.
    """
    summary = {
        "intraday_total_filled": 0,
        "intraday_win_rate": "N/A",
        "opening_total_filled": 0,
        "opening_accuracy": "N/A",
    }

    try:
        if os.path.exists(INTRADAY_LOG_FILE):
            df = pd.read_csv(INTRADAY_LOG_FILE)
            filled = df[df.get("actual_result_manual_fill", "").astype(str).str.len() > 0]
            if not filled.empty:
                # User can fill WIN/LOSS manually.
                wins = filled["actual_result_manual_fill"].astype(str).str.upper().eq("WIN").sum()
                total = len(filled)
                summary["intraday_total_filled"] = int(total)
                summary["intraday_win_rate"] = f"{wins / total * 100:.1f}%"
    except Exception:
        pass

    try:
        if os.path.exists(OPENING_SPIKE_LOG_FILE):
            df = pd.read_csv(OPENING_SPIKE_LOG_FILE)
            filled = df[df.get("actual_opening_spike_manual_fill", "").astype(str).str.len() > 0]
            if not filled.empty:
                correct = 0
                total = len(filled)
                for _, row in filled.iterrows():
                    predicted = str(row.get("bias", "")).replace("SPIKE ", "").upper()
                    actual = str(row.get("actual_opening_spike_manual_fill", "")).upper()
                    if predicted in actual:
                        correct += 1
                summary["opening_total_filled"] = int(total)
                summary["opening_accuracy"] = f"{correct / total * 100:.1f}%"
    except Exception:
        pass

    return summary


# =============================================================================
# 15. MAIN BOT LOOP
# =============================================================================

def update_dashboard_raw(bundle: Dict[str, Any], headlines: List[Dict[str, Any]], news_score: int, data_quality: Dict[str, Any]) -> None:
    df15 = bundle.get("df15")
    df1h = bundle.get("df1h")
    price = bundle.get("price", 0.0)

    latest15 = df15.iloc[-1] if df15 is not None and not df15.empty else None
    latest1h = df1h.iloc[-1] if df1h is not None and not df1h.empty else None

    DASHBOARD_DATA["price"] = price
    DASHBOARD_DATA["news"] = headlines
    DASHBOARD_DATA["news_score"] = news_score
    DASHBOARD_DATA["last_update"] = utc_now_str()
    DASHBOARD_DATA["data_quality"] = data_quality

    if latest15 is not None:
        DASHBOARD_DATA["rsi15"] = safe_float(latest15.get("RSI"))
        DASHBOARD_DATA["atr15"] = safe_float(latest15.get("ATR"))
        DASHBOARD_DATA["ema_status_15m"] = "BULLISH 21>50" if safe_float(latest15.get("EMA21")) > safe_float(latest15.get("EMA50")) else "BEARISH 21<50"

    if latest1h is not None:
        DASHBOARD_DATA["rsi1h"] = safe_float(latest1h.get("RSI"))
        DASHBOARD_DATA["atr1h"] = safe_float(latest1h.get("ATR"))
        DASHBOARD_DATA["ema_status_1h"] = "BULLISH 21>50" if safe_float(latest1h.get("EMA21")) > safe_float(latest1h.get("EMA50")) else "BEARISH 21<50"

    if latest1h is not None:
        DASHBOARD_DATA["trend"] = "BULLISH 🟢" if safe_float(latest1h.get("MACD_HIST")) > 0 else "BEARISH 🔴"


def run_bot() -> None:
    global DASHBOARD_DATA

    last_full_report_time = 0.0
    last_price = 0.0
    seen_news_links: set = set()

    while True:
        try:
            DASHBOARD_DATA["status"] = "Fetching market data..."
            bundle = get_market_bundle()
            df15 = bundle.get("df15")
            df1h = bundle.get("df1h")
            price = bundle.get("price", 0.0)

            headlines, raw_entries, news_score = get_news()
            data_quality = check_data_quality(df15, df1h)

            update_dashboard_raw(bundle, headlines, news_score, data_quality)

            if df15 is None or df15.empty or df1h is None or df1h.empty:
                DASHBOARD_DATA["status"] = "Waiting for enough market data..."
                time.sleep(LOOP_SLEEP_SECONDS)
                continue

            DASHBOARD_DATA["status"] = "Calculating 30-min and opening-spike engines..."

            intraday = calculate_intraday_alert(df15, df1h, news_score, data_quality, last_price)
            opening = calculate_opening_spike_probability(df15, df1h, news_score, data_quality)

            DASHBOARD_DATA["intraday"] = asdict(intraday)
            DASHBOARD_DATA["opening_spike"] = asdict(opening)
            DASHBOARD_DATA["factor_table"] = [asdict(f) for f in intraday.factors]
            DASHBOARD_DATA["opening_factor_table"] = [asdict(f) for f in opening.factors]
            DASHBOARD_DATA["calibration"] = get_calibration_summary()

            # Emergency triggers
            current_time = time.time()
            is_time_up = (current_time - last_full_report_time) >= MAIN_REPORT_INTERVAL_SECONDS
            alert_reason = "30-minute scheduled report"
            is_emergency = False

            # Fast spike checker
            if last_price > 0 and abs(price - last_price) >= FAST_SPIKE_THRESHOLD:
                diff = price - last_price
                is_emergency = True
                alert_reason = f"FAST PRICE SPIKE {diff:+.2f}"

            # Breaking news checker
            current_utc = calendar.timegm(time.gmtime())
            for entry in raw_entries:
                if "published_parsed" in entry:
                    age_seconds = current_utc - calendar.timegm(entry.published_parsed)
                    link = entry.get("link", "")
                    if age_seconds < FRESH_NEWS_SECONDS and link not in seen_news_links:
                        is_emergency = True
                        alert_reason = f"BREAKING NEWS: {clean_title(entry.get('title', ''))}"
                        seen_news_links.add(link)
                        break

            # Opening spike watch can trigger a report even if not 30 min yet.
            if opening.active and opening.confidence >= 55:
                if (current_time - last_full_report_time) >= 900:
                    is_emergency = True
                    alert_reason = "OPENING SPIKE WATCH ACTIVE"

            if is_time_up or is_emergency:
                chart = create_chart(df1h, df15)
                DASHBOARD_DATA["status"] = "Generating explanation..."
                explanation = ai_explain_result(intraday, opening, headlines, data_quality, chart)
                DASHBOARD_DATA["analysis"] = explanation

                log_intraday_result(intraday, news_score, data_quality)
                log_opening_spike_result(opening, price, news_score)

                should_send = True
                if MUTE_WAIT_SIGNALS and intraday.action == "STRICT WAIT" and not is_emergency and not opening.active:
                    should_send = False

                if should_send:
                    telegram_text = build_telegram_text(alert_reason, intraday, opening, data_quality, news_score)
                    send_telegram_message(telegram_text, chart)

                last_full_report_time = time.time()

            last_price = price
            DASHBOARD_DATA["status"] = "Monitoring active"

        except Exception as e:
            DASHBOARD_DATA["status"] = f"Error: {e}"
            print(f"Bot error: {e}")

        time.sleep(LOOP_SLEEP_SECONDS)


# =============================================================================
# 16. DASHBOARD HTML
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WTI COMBINED QUANT TERMINAL</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap" rel="stylesheet">
    <style>
        body { background-color: #070b12; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
        .panel { background: rgba(15, 23, 42, 0.88); border: 1px solid #1e293b; border-radius: 12px; }
        .green { color: #4ade80; text-shadow: 0 0 8px rgba(74,222,128,0.35); }
        .red { color: #f87171; text-shadow: 0 0 8px rgba(248,113,113,0.35); }
        .yellow { color: #facc15; text-shadow: 0 0 8px rgba(250,204,21,0.35); }
        .blue { color: #60a5fa; text-shadow: 0 0 8px rgba(96,165,250,0.35); }
        .muted { color: #94a3b8; }
        .blink { animation: blinker 1.5s linear infinite; }
        @keyframes blinker { 50% { opacity: 0.3; } }
        table { width: 100%; border-collapse: collapse; }
        th, td { border-bottom: 1px solid #1e293b; padding: 8px; text-align: left; font-size: 12px; }
        th { color: #94a3b8; font-weight: 700; text-transform: uppercase; }
    </style>
</head>

<body class="p-5">
<div class="max-w-7xl mx-auto space-y-5">

    <header class="flex flex-col md:flex-row md:items-end justify-between gap-3 pb-4 border-b border-slate-800">
        <div>
            <h1 class="text-2xl md:text-3xl font-extrabold tracking-wider">
                WTI / USOIL COMBINED TERMINAL
                <span class="text-xs green align-top blink">● LIVE</span>
            </h1>
            <p class="muted text-sm mt-1">30-minute alert system + weekend opening-spike probability module</p>
        </div>
        <div class="text-sm muted">
            Status: <span class="yellow">{{ data.status }}</span><br>
            Last update: {{ data.last_update }}
        </div>
    </header>

    <section class="grid grid-cols-1 md:grid-cols-4 gap-5">
        <div class="panel p-5">
            <p class="muted text-xs uppercase mb-2">Spot Price</p>
            <h2 class="text-5xl font-extrabold {% if 'BULLISH' in data.trend %}green{% else %}red{% endif %}">
                ${{ "%.2f"|format(data.price) }}
            </h2>
            <p class="muted text-xs mt-2">{{ data.trend }}</p>
        </div>

        <div class="panel p-5 md:col-span-2">
            {% set intraday = data.intraday %}
            <p class="muted text-xs uppercase mb-3">Primary 30-Minute Alert System</p>
            <div class="flex justify-between items-start">
                <div>
                    <p class="muted text-xs">ACTION</p>
                    <p class="text-4xl font-extrabold tracking-wider
                    {% if intraday.action == 'BUY' %}green{% elif intraday.action == 'SELL' %}red{% else %}yellow{% endif %}">
                        {{ intraday.action or 'WAITING' }}
                    </p>
                </div>
                <div class="text-right">
                    <p class="muted text-xs">CONVICTION</p>
                    <p class="text-3xl font-extrabold yellow">{{ intraday.conviction or 0 }}%</p>
                    <p class="muted text-xs">Score: {{ intraday.score or 0 }}</p>
                </div>
            </div>

            <div class="grid grid-cols-3 gap-3 mt-5 pt-4 border-t border-slate-800 text-center">
                <div>
                    <p class="muted text-xs">ENTRY</p>
                    <p class="text-lg">${{ "%.2f"|format(intraday.entry or 0) }}</p>
                </div>
                <div>
                    <p class="muted text-xs">TARGET</p>
                    <p class="text-lg green">${{ "%.2f"|format(intraday.target or 0) }}</p>
                </div>
                <div>
                    <p class="muted text-xs">STOP</p>
                    <p class="text-lg red">${{ "%.2f"|format(intraday.stop or 0) }}</p>
                </div>
            </div>
            <p class="text-xs muted mt-4">{{ intraday.reason }}</p>
        </div>

        <div class="panel p-5">
            <p class="muted text-xs uppercase mb-3">Technical HUD</p>
            <div class="space-y-3 text-sm">
                <div class="flex justify-between"><span class="muted">15m RSI</span><span>{{ "%.2f"|format(data.rsi15) }}</span></div>
                <div class="flex justify-between"><span class="muted">1H RSI</span><span>{{ "%.2f"|format(data.rsi1h) }}</span></div>
                <div class="flex justify-between"><span class="muted">15m ATR</span><span>${{ "%.2f"|format(data.atr15) }}</span></div>
                <div class="flex justify-between"><span class="muted">1H ATR</span><span>${{ "%.2f"|format(data.atr1h) }}</span></div>
                <div class="flex justify-between"><span class="muted">15m EMA</span><span class="{% if 'BULLISH' in data.ema_status_15m %}green{% else %}red{% endif %}">{{ data.ema_status_15m }}</span></div>
                <div class="flex justify-between"><span class="muted">1H EMA</span><span class="{% if 'BULLISH' in data.ema_status_1h %}green{% else %}red{% endif %}">{{ data.ema_status_1h }}</span></div>
            </div>
        </div>
    </section>

    <section class="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div class="panel p-5 lg:col-span-2">
            {% set opening = data.opening_spike %}
            <div class="flex justify-between items-start">
                <div>
                    <p class="muted text-xs uppercase mb-2">Weekend Opening-Spike Module</p>
                    <h2 class="text-3xl font-extrabold
                    {% if opening.bias == 'SPIKE UP' %}green{% elif opening.bias == 'SPIKE DOWN' %}red{% else %}yellow{% endif %}">
                        {{ opening.bias or 'WAITING' }}
                    </h2>
                    <p class="muted text-xs mt-1">Active watch window: {{ opening.active }}</p>
                </div>
                <div class="text-right">
                    <p class="muted text-xs">CONFIDENCE</p>
                    <p class="text-2xl font-extrabold yellow">{{ opening.confidence or 0 }}%</p>
                </div>
            </div>

            <div class="grid grid-cols-3 gap-3 mt-5 text-center">
                <div class="bg-slate-900 rounded-lg p-4">
                    <p class="muted text-xs">SPIKE UP</p>
                    <p class="text-3xl font-extrabold green">{{ opening.up_probability or 0 }}%</p>
                </div>
                <div class="bg-slate-900 rounded-lg p-4">
                    <p class="muted text-xs">SPIKE DOWN</p>
                    <p class="text-3xl font-extrabold red">{{ opening.down_probability or 0 }}%</p>
                </div>
                <div class="bg-slate-900 rounded-lg p-4">
                    <p class="muted text-xs">WHIPSAW</p>
                    <p class="text-3xl font-extrabold yellow">{{ opening.whipsaw_probability or 0 }}%</p>
                </div>
            </div>

            <p class="text-xs muted mt-4">{{ opening.reason }}</p>
            <p class="text-xs muted mt-2">{{ opening.historical_summary }}</p>
        </div>

        <div class="panel p-5">
            <p class="muted text-xs uppercase mb-3">Data Quality / Calibration</p>
            <div class="space-y-3 text-sm">
                <div class="flex justify-between"><span class="muted">Data status</span><span class="{% if data.data_quality.status == 'GOOD' %}green{% elif data.data_quality.status == 'CAUTION' %}yellow{% else %}red{% endif %}">{{ data.data_quality.status }}</span></div>
                <div class="flex justify-between"><span class="muted">Penalty</span><span>{{ data.data_quality.penalty }}</span></div>
                <div class="flex justify-between"><span class="muted">News score</span><span class="{% if data.news_score > 0 %}green{% elif data.news_score < 0 %}red{% else %}muted{% endif %}">{{ data.news_score }}</span></div>
                <div class="flex justify-between"><span class="muted">Intraday filled</span><span>{{ data.calibration.intraday_total_filled or 0 }}</span></div>
                <div class="flex justify-between"><span class="muted">Intraday win rate</span><span>{{ data.calibration.intraday_win_rate or 'N/A' }}</span></div>
                <div class="flex justify-between"><span class="muted">Opening filled</span><span>{{ data.calibration.opening_total_filled or 0 }}</span></div>
                <div class="flex justify-between"><span class="muted">Opening accuracy</span><span>{{ data.calibration.opening_accuracy or 'N/A' }}</span></div>
            </div>
            {% if data.data_quality.warnings %}
            <div class="mt-4 text-xs red">
                {% for w in data.data_quality.warnings %}
                <p>⚠ {{ w }}</p>
                {% endfor %}
            </div>
            {% endif %}
        </div>
    </section>

    <section class="panel p-5">
        <p class="muted text-xs uppercase mb-3">AI Explanation Log</p>
        <div class="text-sm leading-relaxed">{{ data.analysis | safe }}</div>
    </section>

    <section class="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div class="panel p-5">
            <p class="muted text-xs uppercase mb-3">30-Minute Factor Table</p>
            <table>
                <thead>
                    <tr><th>Factor</th><th>Signal</th><th>Weight</th><th>Note</th></tr>
                </thead>
                <tbody>
                {% for f in data.factor_table %}
                    <tr>
                        <td>{{ f.name }}</td>
                        <td>{{ f.signal }}</td>
                        <td class="{% if f.weight > 0 %}green{% elif f.weight < 0 %}red{% else %}muted{% endif %}">{{ f.weight }}</td>
                        <td class="muted">{{ f.note }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="panel p-5">
            <p class="muted text-xs uppercase mb-3">Opening-Spike Factor Table</p>
            <table>
                <thead>
                    <tr><th>Factor</th><th>Signal</th><th>Weight</th><th>Note</th></tr>
                </thead>
                <tbody>
                {% for f in data.opening_factor_table %}
                    <tr>
                        <td>{{ f.name }}</td>
                        <td>{{ f.signal }}</td>
                        <td class="{% if f.weight > 0 %}green{% elif f.weight < 0 %}red{% else %}muted{% endif %}">{{ f.weight }}</td>
                        <td class="muted">{{ f.note }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
    </section>

    <section class="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div class="panel p-5 lg:col-span-2">
            <p class="muted text-xs uppercase mb-3">Chart: 1H Structure with Manual Levels</p>
            <img src="/chart" class="w-full rounded-lg border border-slate-800" alt="WTI chart">
        </div>

        <div class="panel p-5 max-h-[650px] overflow-y-auto">
            <p class="muted text-xs uppercase mb-3">48-Hour Scored News Feed</p>
            <div class="space-y-4">
                {% for item in data.news %}
                    <div class="border-l-2 {% if item.score > 0 %}border-green-500{% elif item.score < 0 %}border-red-500{% else %}border-slate-500{% endif %} pl-3">
                        <p class="text-xs muted">{{ item.time }} | age {{ item.age_hours }}h</p>
                        <a href="{{ item.link }}" target="_blank" class="text-sm hover:text-white">{{ item.title }}</a>
                        <p class="text-xs mt-1 {% if item.score > 0 %}green{% elif item.score < 0 %}red{% else %}muted{% endif %}">
                            {{ item.direction }} | score {{ item.score }} | {{ item.hits }}
                        </p>
                    </div>
                {% endfor %}
                {% if not data.news %}
                    <p class="muted text-sm">No fresh news loaded yet.</p>
                {% endif %}
            </div>
        </div>
    </section>

</div>

<script>
    setTimeout(function(){ location.reload(); }, 60000);
</script>

</body>
</html>
"""


# =============================================================================
# 17. WEB ROUTES
# =============================================================================

@app.route("/")
def dashboard():
    return render_template_string(HTML_TEMPLATE, data=DASHBOARD_DATA)


@app.route("/chart")
def serve_chart():
    if os.path.exists(CHART_FILE):
        return send_file(CHART_FILE, mimetype="image/png")
    return "Chart generating. Refresh shortly.", 404


@app.route("/health")
def health():
    return {
        "status": DASHBOARD_DATA.get("status"),
        "last_update": DASHBOARD_DATA.get("last_update"),
        "price": DASHBOARD_DATA.get("price"),
        "data_quality": DASHBOARD_DATA.get("data_quality"),
    }


# =============================================================================
# 18. STARTUP
# =============================================================================

if __name__ == "__main__":
    print("Starting WTI Combined 30-Min + Opening-Spike Bot v3...")
    print(f"Ticker: {TICKER}")
    print(f"Dashboard: http://localhost:{PORT}")
    print("Decision rule: deterministic model first, Gemini explanation second.")

    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    app.run(host="0.0.0.0", port=PORT)
