"""
WTI / USOIL Opening-Spike Probability Bot v2
============================================

Purpose
-------
This version upgrades the original prototype into a more defensible decision-support
system for the FIRST reopening spike / gap direction in WTI crude oil.

Key design change
-----------------
The AI model is NOT allowed to decide BUY/SELL or invent probabilities.
The deterministic probability engine calculates:
    - Spike UP probability
    - Spike DOWN probability
    - WHIPSAW / no-clean-spike probability

Gemini, if configured, is used only to explain the already-calculated result.

Main upgrades
-------------
1. Rules-based probability engine with visible factor weights.
2. News classification score instead of raw AI interpretation.
3. Historical weekend opening-spike extraction and calibration support.
4. Data-quality checks and signal blocking.
5. Prediction audit logging for future calibration.
6. Dashboard shows evidence table, not only AI commentary.
7. Telegram alerts use model output, not AI opinion.

Environment variables
---------------------
Required only if you want Telegram/Gemini:
    GEMINI_API_KEY
    TELEGRAM_TOKEN
    TELEGRAM_CHAT_ID

Optional:
    PORT=8080
    TICKER=CL=F
    REPORT_INTERVAL_SECONDS=1800
    LOOP_SLEEP_SECONDS=120
    MIN_SIGNAL_CONFIDENCE=58
    MIN_DATA_QUALITY=60

Run
---
    pip install flask yfinance requests feedparser pandas numpy mplfinance ta
    python wti_opening_spike_bot_v2.py

Open dashboard
--------------
    http://localhost:8080
"""

from __future__ import annotations

import base64
import calendar
import csv
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, render_template_string, send_file
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange


# =============================================================================
# 1. CONFIGURATION
# =============================================================================


@dataclass
class TechnicalLevels:
    """Manual levels from your chart / thesis.

    Update these whenever your chart levels change. The model is transparent:
    every probability factor that uses these levels appears on the dashboard.
    """

    support_low: float = 94.33
    support_high: float = 94.90
    reclaim_1: float = 95.57
    reclaim_2_low: float = 96.40
    reclaim_2_high: float = 96.70
    resistance_1_low: float = 98.53
    resistance_1_high: float = 98.67
    resistance_2: float = 100.45
    downside_1_high: float = 91.50
    downside_1_low: float = 90.00
    downside_2: float = 86.71


@dataclass
class BotConfig:
    ticker: str = os.environ.get("TICKER", "CL=F")
    port: int = int(os.environ.get("PORT", "8080"))

    # Timeframes
    intraday_period: str = "60d"
    intraday_interval: str = "15m"
    hourly_period: str = "60d"
    hourly_interval: str = "1h"

    # Loop timing
    loop_sleep_seconds: int = int(os.environ.get("LOOP_SLEEP_SECONDS", "120"))
    report_interval_seconds: int = int(os.environ.get("REPORT_INTERVAL_SECONDS", "1800"))

    # Signal gates
    min_signal_confidence: float = float(os.environ.get("MIN_SIGNAL_CONFIDENCE", "58"))
    min_data_quality: float = float(os.environ.get("MIN_DATA_QUALITY", "60"))
    mute_blocked_signals: bool = os.environ.get("MUTE_BLOCKED_SIGNALS", "true").lower() == "true"

    # External keys
    gemini_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
    telegram_token: Optional[str] = os.environ.get("TELEGRAM_TOKEN")
    telegram_chat_id: Optional[str] = os.environ.get("TELEGRAM_CHAT_ID")

    # Files
    data_dir: Path = Path(os.environ.get("BOT_DATA_DIR", "."))
    chart_file: str = "oil_chart_v2.png"
    weekend_history_csv: str = "weekend_opening_history.csv"
    prediction_log_csv: str = "prediction_audit_log.csv"

    levels: TechnicalLevels = field(default_factory=TechnicalLevels)


CONFIG = BotConfig()


# =============================================================================
# 2. GLOBAL DASHBOARD STATE
# =============================================================================


DASHBOARD_DATA: Dict[str, Any] = {
    "status": "BOOTING",
    "last_update": "N/A",
    "ticker": CONFIG.ticker,
    "price": 0.0,
    "trend": "WAITING",
    "rsi": 0.0,
    "atr": 0.0,
    "ema_status": "WAITING",
    "data_quality": 0.0,
    "data_warnings": [],
    "news": [],
    "news_score": 0,
    "news_bias": "NEUTRAL",
    "prob_up": 0.0,
    "prob_down": 0.0,
    "prob_whipsaw": 0.0,
    "model_confidence": 0.0,
    "signal_permission": "BLOCKED",
    "signal_direction": "NO TRADE",
    "factor_rows": [],
    "historical_summary": "No history loaded yet.",
    "calibration_summary": "No calibration yet.",
    "explanation": "Awaiting initial calculation...",
    "trade_entry": 0.0,
    "trade_stop": 0.0,
    "trade_target": 0.0,
    "risk_label": "UNKNOWN",
}


app = Flask(__name__)


# =============================================================================
# 3. UTILITY HELPERS
# =============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_str() -> str:
    return utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return fallback
        return float(value)
    except Exception:
        return fallback


def ensure_data_files() -> None:
    """Create CSV files with headers if missing."""
    CONFIG.data_dir.mkdir(parents=True, exist_ok=True)

    history_path = CONFIG.data_dir / CONFIG.weekend_history_csv
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "case_id",
                    "reopen_time_utc",
                    "friday_close",
                    "reopen_open",
                    "first_15m_high",
                    "first_15m_low",
                    "first_15m_close",
                    "gap_points",
                    "first_15m_net_points",
                    "first_15m_range",
                    "spike_direction",
                    "rsi",
                    "atr",
                    "ema_status",
                    "support_distance",
                    "resistance_distance",
                    "news_score",
                    "notes",
                ]
            )

    log_path = CONFIG.data_dir / CONFIG.prediction_log_csv
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_utc",
                    "ticker",
                    "price",
                    "prob_up",
                    "prob_down",
                    "prob_whipsaw",
                    "confidence",
                    "signal_direction",
                    "signal_permission",
                    "news_score",
                    "data_quality",
                    "factor_score",
                    "risk_label",
                    "actual_direction_later",
                    "notes",
                ]
            )


# =============================================================================
# 4. MARKET DATA
# =============================================================================


def normalize_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex and keep OHLCV numeric columns."""
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance may return ('Close', 'CL=F'). Drop ticker level.
        data.columns = data.columns.droplevel(-1)

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in data.columns]
    data = data[keep_cols]

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    return data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, ATR, EMA21 and EMA50."""
    if data.empty or len(data) < 60:
        return data

    data = data.copy()
    close = data["Close"]

    data["RSI"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close)
    data["MACD"] = macd.macd()
    data["Signal"] = macd.macd_signal()
    data["ATR"] = AverageTrueRange(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=14,
    ).average_true_range()
    data["EMA21"] = EMAIndicator(close=close, window=21).ema_indicator()
    data["EMA50"] = EMAIndicator(close=close, window=50).ema_indicator()
    return data


def download_market_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download both 15m and 1h data."""
    intraday = yf.download(
        CONFIG.ticker,
        period=CONFIG.intraday_period,
        interval=CONFIG.intraday_interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    hourly = yf.download(
        CONFIG.ticker,
        period=CONFIG.hourly_period,
        interval=CONFIG.hourly_interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    intraday = add_indicators(normalize_yfinance_columns(intraday))
    hourly = add_indicators(normalize_yfinance_columns(hourly))
    return intraday, hourly


def latest_market_snapshot(intraday: pd.DataFrame, hourly: pd.DataFrame) -> Dict[str, Any]:
    """Return latest values for the probability engine."""
    source = hourly if not hourly.empty and len(hourly) >= 60 else intraday
    if source.empty:
        raise ValueError("No market data available")

    last = source.iloc[-1]
    price = safe_float(last["Close"])
    rsi = safe_float(last.get("RSI"), 50.0)
    atr = safe_float(last.get("ATR"), 0.0)
    ema21 = safe_float(last.get("EMA21"), price)
    ema50 = safe_float(last.get("EMA50"), price)
    macd = safe_float(last.get("MACD"), 0.0)
    signal = safe_float(last.get("Signal"), 0.0)

    ema_status = "BULLISH (21 > 50)" if ema21 > ema50 else "BEARISH (21 < 50)"
    trend = "BULLISH 🟢" if macd > signal else "BEARISH 🔴"

    recent_window = source.tail(250) if len(source) >= 250 else source
    recent_high = safe_float(recent_window["High"].max(), price)
    recent_low = safe_float(recent_window["Low"].min(), price)

    return {
        "price": price,
        "rsi": rsi,
        "atr": atr,
        "ema21": ema21,
        "ema50": ema50,
        "ema_status": ema_status,
        "trend": trend,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "last_timestamp": source.index[-1],
    }


# =============================================================================
# 5. DATA QUALITY CHECKS
# =============================================================================


def assess_data_quality(intraday: pd.DataFrame, hourly: pd.DataFrame, snapshot: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Score data quality from 0 to 100 and return warnings."""
    score = 100.0
    warnings: List[str] = []

    if intraday.empty:
        score -= 35
        warnings.append("15m data unavailable")
    if hourly.empty:
        score -= 35
        warnings.append("1h data unavailable")

    price = snapshot.get("price", 0.0)
    atr = snapshot.get("atr", 0.0)
    rsi = snapshot.get("rsi", 50.0)

    if price <= 0:
        score -= 50
        warnings.append("Invalid latest price")
    if atr <= 0 or math.isnan(atr):
        score -= 20
        warnings.append("ATR unavailable or zero")
    if not 0 <= rsi <= 100:
        score -= 20
        warnings.append("RSI invalid")

    # Cross-check latest 15m and 1h close. This is not a second vendor, but it
    # catches stale/misaligned intervals.
    if not intraday.empty and not hourly.empty:
        last_15m = safe_float(intraday["Close"].iloc[-1])
        last_1h = safe_float(hourly["Close"].iloc[-1])
        if abs(last_15m - last_1h) > max(0.30, atr * 0.75):
            score -= 20
            warnings.append(f"15m/1h close mismatch: {last_15m:.2f} vs {last_1h:.2f}")

    # Staleness warning. Futures have weekend closures; this is a warning, not a hard fail.
    last_ts = snapshot.get("last_timestamp")
    try:
        if hasattr(last_ts, "tz_convert"):
            last_ts_utc = last_ts.tz_convert("UTC").to_pydatetime()
        elif hasattr(last_ts, "to_pydatetime"):
            last_ts_utc = last_ts.to_pydatetime()
            if last_ts_utc.tzinfo is None:
                last_ts_utc = last_ts_utc.replace(tzinfo=timezone.utc)
        else:
            last_ts_utc = utc_now()

        age_hours = (utc_now() - last_ts_utc).total_seconds() / 3600
        if age_hours > 72:
            score -= 25
            warnings.append(f"Data appears stale: latest bar is {age_hours:.1f}h old")
        elif age_hours > 30:
            score -= 10
            warnings.append(f"Weekend/stale-feed warning: latest bar is {age_hours:.1f}h old")
    except Exception:
        score -= 5
        warnings.append("Could not verify data timestamp freshness")

    return clamp(score, 0, 100), warnings


# =============================================================================
# 6. CHARTING
# =============================================================================


def calculate_universal_channel(slice_data: pd.DataFrame, cutoff_pct: float = 0.92) -> Tuple[bool, List[Tuple[Any, float]], List[Tuple[Any, float]]]:
    """Linear regression channel using percentile high/low offsets."""
    if slice_data is None or len(slice_data) < 20:
        return False, [], []

    x = np.arange(len(slice_data))
    y = slice_data["Close"].values.astype(float)
    m, b = np.polyfit(x, y, 1)
    reg_line = m * x + b

    high_offsets = slice_data["High"].values.astype(float) - reg_line
    low_offsets = reg_line - slice_data["Low"].values.astype(float)

    sorted_highs = np.sort(high_offsets)
    sorted_lows = np.sort(low_offsets)
    cutoff_idx = int(len(sorted_highs) * cutoff_pct)
    cutoff_idx = min(max(cutoff_idx, 0), len(sorted_highs) - 1)

    upper_offset = sorted_highs[cutoff_idx]
    lower_offset = sorted_lows[cutoff_idx]

    upper_channel = reg_line + upper_offset
    lower_channel = reg_line - lower_offset

    date_start = slice_data.index[0]
    date_end = slice_data.index[-1]
    support_line = [(date_start, float(lower_channel[0])), (date_end, float(lower_channel[-1]))]
    resistance_line = [(date_start, float(upper_channel[0])), (date_end, float(upper_channel[-1]))]
    return True, support_line, resistance_line


def create_chart(plot_data: pd.DataFrame, snapshot: Dict[str, Any]) -> Optional[str]:
    """Create chart with macro/micro regression channels and manual levels."""
    if plot_data is None or plot_data.empty:
        return None

    fname = CONFIG.chart_file
    levels = CONFIG.levels

    macro_exists, macro_sup, macro_res = calculate_universal_channel(plot_data, cutoff_pct=0.92)
    micro_exists, micro_sup, micro_res = calculate_universal_channel(plot_data.tail(70), cutoff_pct=0.80)

    mc = mpf.make_marketcolors(
        up="#00E676",
        down="#D500F9",
        edge="inherit",
        wick="inherit",
        volume="in",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor="#0f172a",
        edgecolor="#1e293b",
        figcolor="#0f172a",
    )

    hlines = [
        levels.support_low,
        levels.support_high,
        levels.reclaim_1,
        levels.reclaim_2_low,
        levels.reclaim_2_high,
        levels.resistance_1_low,
        levels.resistance_1_high,
        levels.resistance_2,
    ]

    kwargs: Dict[str, Any] = dict(
        type="candle",
        style=style,
        volume=False,
        mav=(21, 50),
        hlines=dict(
            hlines=hlines,
            colors=["#fbbf24"] * len(hlines),
            linestyle="--",
            linewidths=0.8,
        ),
        savefig=dict(fname=fname, dpi=130, bbox_inches="tight"),
    )

    lines_to_draw: List[List[Tuple[Any, float]]] = []
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
    return fname


# =============================================================================
# 7. NEWS INGESTION AND SCORING
# =============================================================================


NEWS_KEYWORDS: List[Tuple[str, int, str]] = [
    # Bullish/supply-risk keywords
    (r"\b(strike|strikes|attack|attacks|attacked|missile|drone|explosion|blast)\b", 18, "military attack / strike risk"),
    (r"\b(tanker|shipping|vessel|maritime|red sea|hormuz|strait of hormuz)\b", 16, "shipping / chokepoint risk"),
    (r"\b(iran|israel|houthis|yemen|middle east|gulf)\b", 12, "Middle East risk"),
    (r"\b(sanction|sanctions|embargo|export ban)\b", 14, "sanctions / export restriction"),
    (r"\b(opec cut|output cut|production cut|supply cut|cuts production)\b", 18, "supply cut"),
    (r"\b(inventory draw|stockpile draw|drawdown|crude draw)\b", 12, "inventory draw"),
    (r"\b(refinery outage|pipeline outage|terminal outage|force majeure)\b", 15, "infrastructure outage"),

    # Bearish/de-escalation or demand weakness keywords
    (r"\b(ceasefire|truce|peace talks|deal reached|diplomatic breakthrough|de-escalation|deescalation)\b", -25, "de-escalation"),
    (r"\b(output increase|raise production|production increase|opec increase|boost output)\b", -20, "supply increase"),
    (r"\b(inventory build|stockpile build|crude build)\b", -15, "inventory build"),
    (r"\b(demand weak|weak demand|slowdown|recession|growth concerns)\b", -15, "demand weakness"),
    (r"\b(dollar strengthens|strong dollar|rate hike|higher rates)\b", -8, "macro pressure"),
]


def fetch_news() -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Fetch Google News RSS headlines with a 48-hour freshness filter.

    Google News RSS is not perfect. The scoring engine therefore uses it as a
    catalyst input, not as a sole source of truth.
    """
    try:
        query = '("Crude Oil" OR "WTI" OR "OPEC" OR "Middle East" OR "Hormuz" OR "oil tanker") when:2d'
        base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
        final_url = base_url.format(requests.utils.quote(query))
        feed = feedparser.parse(final_url)
        if not feed.entries:
            return [], []

        current_time = calendar.timegm(time.gmtime())
        headlines: List[Dict[str, Any]] = []
        raw_entries: List[Any] = []

        for entry in feed.entries:
            if "published_parsed" not in entry:
                continue
            entry_time = calendar.timegm(entry.published_parsed)
            age_hours = (current_time - entry_time) / 3600
            if age_hours <= 48:
                title = entry.title.split(" - ")[0].strip()
                headlines.append(
                    {
                        "title": title,
                        "time": entry.get("published", ""),
                        "link": entry.link,
                        "age_hours": round(age_hours, 2),
                    }
                )
                raw_entries.append(entry)
            if len(headlines) >= 10:
                break

        return headlines, raw_entries
    except Exception as exc:
        print(f"News fetch error: {exc}")
        return [], []


def score_headline(title: str) -> Tuple[int, List[str]]:
    """Score a headline based on predefined catalyst keywords."""
    score = 0
    reasons: List[str] = []
    text = title.lower()

    for pattern, weight, reason in NEWS_KEYWORDS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            score += weight
            reasons.append(reason)

    return score, sorted(set(reasons))


def score_news(headlines: List[Dict[str, Any]]) -> Tuple[int, str, List[Dict[str, Any]]]:
    """Return total news score, bias label, and scored headline list."""
    total = 0
    scored: List[Dict[str, Any]] = []

    for item in headlines:
        headline_score, reasons = score_headline(item["title"])
        # More recent items matter more.
        age_hours = float(item.get("age_hours", 24.0))
        freshness_multiplier = 1.0 if age_hours <= 6 else 0.75 if age_hours <= 24 else 0.50
        adjusted = int(round(headline_score * freshness_multiplier))
        total += adjusted
        scored.append({**item, "score": adjusted, "reasons": ", ".join(reasons) if reasons else "neutral"})

    total = int(clamp(total, -60, 60))
    if total >= 25:
        bias = "BULLISH SUPPLY-RISK"
    elif total <= -25:
        bias = "BEARISH DE-ESCALATION / DEMAND"
    elif total > 5:
        bias = "MILD BULLISH"
    elif total < -5:
        bias = "MILD BEARISH"
    else:
        bias = "NEUTRAL"

    return total, bias, scored


# =============================================================================
# 8. HISTORICAL WEEKEND OPENING CASES
# =============================================================================


def classify_spike(open_price: float, high: float, low: float, close: float, atr: float = 0.0) -> str:
    """Classify the first 15m opening spike.

    The first bar may wick both ways. We classify by dominant excursion and close.
    """
    up_excursion = high - open_price
    down_excursion = open_price - low
    net = close - open_price
    threshold = max(0.08, atr * 0.15 if atr > 0 else 0.08)

    if up_excursion > down_excursion * 1.25 and net > -threshold:
        return "UP"
    if down_excursion > up_excursion * 1.25 and net < threshold:
        return "DOWN"
    if net > threshold and up_excursion >= down_excursion:
        return "UP"
    if net < -threshold and down_excursion >= up_excursion:
        return "DOWN"
    return "WHIPSAW"


def extract_weekend_cases_from_intraday(intraday: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract weekend reopening cases by detecting large time gaps.

    This works best with continuous 15m futures data. It is intentionally simple
    and auditable. You can manually correct the generated CSV if your broker's
    session timing differs.
    """
    if intraday.empty or len(intraday) < 100:
        return []

    df = intraday.copy()
    df = df.sort_index()
    cases: List[Dict[str, Any]] = []

    for i in range(1, len(df)):
        prev_ts = df.index[i - 1]
        curr_ts = df.index[i]
        try:
            delta_hours = (curr_ts - prev_ts).total_seconds() / 3600
        except Exception:
            continue

        # Weekend or holiday gap. For WTI, a normal intraday gap is small;
        # weekend gaps are much larger.
        if delta_hours < 30:
            continue

        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]
        friday_close = safe_float(prev_row["Close"])
        reopen_open = safe_float(curr_row["Open"])
        high = safe_float(curr_row["High"])
        low = safe_float(curr_row["Low"])
        close = safe_float(curr_row["Close"])
        atr = safe_float(curr_row.get("ATR"), 0.0)
        rsi = safe_float(curr_row.get("RSI"), 50.0)
        ema21 = safe_float(curr_row.get("EMA21"), close)
        ema50 = safe_float(curr_row.get("EMA50"), close)
        ema_status = "BULLISH" if ema21 > ema50 else "BEARISH"

        direction = classify_spike(reopen_open, high, low, close, atr=atr)
        case_id = curr_ts.strftime("%Y-%m-%d_%H%M") if hasattr(curr_ts, "strftime") else str(curr_ts)

        levels = CONFIG.levels
        support_distance = reopen_open - levels.support_high
        resistance_distance = levels.resistance_1_low - reopen_open

        cases.append(
            {
                "case_id": case_id,
                "reopen_time_utc": str(curr_ts),
                "friday_close": round(friday_close, 4),
                "reopen_open": round(reopen_open, 4),
                "first_15m_high": round(high, 4),
                "first_15m_low": round(low, 4),
                "first_15m_close": round(close, 4),
                "gap_points": round(reopen_open - friday_close, 4),
                "first_15m_net_points": round(close - reopen_open, 4),
                "first_15m_range": round(high - low, 4),
                "spike_direction": direction,
                "rsi": round(rsi, 2),
                "atr": round(atr, 4),
                "ema_status": ema_status,
                "support_distance": round(support_distance, 4),
                "resistance_distance": round(resistance_distance, 4),
                "news_score": 0,
                "notes": "auto-extracted; manually verify session timing",
            }
        )

    return cases


def upsert_weekend_history(cases: List[Dict[str, Any]]) -> None:
    """Insert extracted cases into CSV without duplicating case_id."""
    if not cases:
        return

    path = CONFIG.data_dir / CONFIG.weekend_history_csv
    ensure_data_files()

    existing_ids = set()
    if path.exists():
        try:
            existing = pd.read_csv(path)
            if "case_id" in existing.columns:
                existing_ids = set(existing["case_id"].astype(str).tolist())
        except Exception:
            existing_ids = set()

    new_cases = [c for c in cases if str(c.get("case_id")) not in existing_ids]
    if not new_cases:
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_cases[0].keys()))
        for case in new_cases:
            writer.writerow(case)


def load_weekend_history() -> pd.DataFrame:
    path = CONFIG.data_dir / CONFIG.weekend_history_csv
    ensure_data_files()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def summarize_history(history: pd.DataFrame, current_news_score: int) -> Tuple[str, Dict[str, float]]:
    """Summarize historical opening spike distribution.

    Also returns a small probability adjustment from similar cases.
    """
    if history.empty or "spike_direction" not in history.columns:
        return "No historical weekend cases available yet.", {"up": 0.0, "down": 0.0, "whipsaw": 0.0, "match_strength": 0.0}

    df = history.dropna(subset=["spike_direction"]).copy()
    if df.empty:
        return "Historical file exists, but no labeled cases found.", {"up": 0.0, "down": 0.0, "whipsaw": 0.0, "match_strength": 0.0}

    df["spike_direction"] = df["spike_direction"].astype(str).str.upper()
    total = len(df)
    up_count = int((df["spike_direction"] == "UP").sum())
    down_count = int((df["spike_direction"] == "DOWN").sum())
    whip_count = int((df["spike_direction"] == "WHIPSAW").sum())

    # Similarity idea: if current news is strongly bullish, compare to cases
    # with positive news_score if available. If no scored cases exist, use all.
    similar = df
    if "news_score" in df.columns:
        numeric_news = pd.to_numeric(df["news_score"], errors="coerce").fillna(0)
        if current_news_score >= 20:
            similar = df[numeric_news >= 10]
        elif current_news_score <= -20:
            similar = df[numeric_news <= -10]

    if len(similar) >= 3:
        sim_up = (similar["spike_direction"] == "UP").mean()
        sim_down = (similar["spike_direction"] == "DOWN").mean()
        sim_whip = (similar["spike_direction"] == "WHIPSAW").mean()
        match_strength = min(1.0, len(similar) / 10.0)
    else:
        sim_up = up_count / total
        sim_down = down_count / total
        sim_whip = whip_count / total
        match_strength = min(0.5, total / 20.0)

    # Translate historical distribution into small adjustment, not full control.
    adj = {
        "up": round((sim_up - 1 / 3) * 18 * match_strength, 2),
        "down": round((sim_down - 1 / 3) * 18 * match_strength, 2),
        "whipsaw": round((sim_whip - 1 / 3) * 12 * match_strength, 2),
        "match_strength": round(match_strength * 100, 1),
    }

    summary = (
        f"Loaded {total} weekend cases: UP {up_count}, DOWN {down_count}, "
        f"WHIPSAW {whip_count}. Historical match strength: {adj['match_strength']}%."
    )
    return summary, adj


# =============================================================================
# 9. PROBABILITY ENGINE
# =============================================================================


def add_factor(rows: List[Dict[str, Any]], name: str, signal: str, weight: float, reason: str) -> None:
    rows.append(
        {
            "name": name,
            "signal": signal,
            "weight": round(weight, 2),
            "reason": reason,
        }
    )


def technical_factor_score(snapshot: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculate transparent technical score.

    Positive score = higher probability of first spike up.
    Negative score = higher probability of first spike down.
    """
    levels = CONFIG.levels
    price = float(snapshot["price"])
    rsi = float(snapshot["rsi"])
    atr = max(float(snapshot["atr"]), 0.01)
    ema21 = float(snapshot["ema21"])
    ema50 = float(snapshot["ema50"])
    recent_high = float(snapshot["recent_high"])
    recent_low = float(snapshot["recent_low"])

    rows: List[Dict[str, Any]] = []
    score = 0.0

    # Support zone logic
    if levels.support_low <= price <= levels.support_high:
        weight = 14
        score += weight
        add_factor(rows, "Manual support zone", "BULLISH", weight, f"Price is inside {levels.support_low:.2f}-{levels.support_high:.2f} support.")
    elif price < levels.support_low:
        weight = -18
        score += weight
        add_factor(rows, "Support lost", "BEARISH", weight, f"Price is below {levels.support_low:.2f}; support failed.")
    elif 0 < price - levels.support_high <= atr * 0.8:
        weight = 7
        score += weight
        add_factor(rows, "Near support", "MILD BULLISH", weight, "Price is still close to support, so dip-buy reaction is possible.")
    else:
        add_factor(rows, "Manual support zone", "NEUTRAL", 0, "Price is not directly interacting with support.")

    # Reclaim levels
    if price >= levels.reclaim_2_high:
        weight = 16
        score += weight
        add_factor(rows, "Major reclaim", "BULLISH", weight, f"Price is above {levels.reclaim_2_high:.2f} reclaim band.")
    elif price >= levels.reclaim_1:
        weight = 8
        score += weight
        add_factor(rows, "First reclaim", "MILD BULLISH", weight, f"Price is above {levels.reclaim_1:.2f}.")
    elif price < levels.reclaim_1:
        weight = -7
        score += weight
        add_factor(rows, "Reclaim missing", "MILD BEARISH", weight, f"Price is still below {levels.reclaim_1:.2f}.")

    # Resistance proximity
    dist_to_res = levels.resistance_1_low - price
    if 0 <= dist_to_res <= atr * 1.2:
        weight = -9
        score += weight
        add_factor(rows, "Resistance overhead", "BEARISH CAP", weight, "Price is close to first resistance/spike target.")
    elif price > levels.resistance_1_high:
        weight = 10
        score += weight
        add_factor(rows, "Resistance cleared", "BULLISH", weight, "Price is above the first resistance band.")
    else:
        add_factor(rows, "Resistance overhead", "NEUTRAL", 0, "Resistance is not immediately overhead.")

    # EMA trend
    if ema21 > ema50:
        weight = 9
        score += weight
        add_factor(rows, "EMA 21/50", "BULLISH", weight, "EMA21 is above EMA50.")
    else:
        weight = -9
        score += weight
        add_factor(rows, "EMA 21/50", "BEARISH", weight, "EMA21 is below EMA50.")

    # RSI momentum
    if rsi >= 60:
        weight = 6
        score += weight
        add_factor(rows, "RSI", "BULLISH", weight, f"RSI {rsi:.1f} shows positive momentum.")
    elif rsi <= 40:
        weight = -6
        score += weight
        add_factor(rows, "RSI", "BEARISH", weight, f"RSI {rsi:.1f} shows negative momentum.")
    else:
        add_factor(rows, "RSI", "NEUTRAL", 0, f"RSI {rsi:.1f} is neutral.")

    # Recent range location
    recent_range = max(recent_high - recent_low, 0.01)
    range_pos = (price - recent_low) / recent_range
    if range_pos <= 0.25:
        weight = 5
        score += weight
        add_factor(rows, "Range position", "BOUNCE RISK", weight, "Price is near the lower quarter of recent range.")
    elif range_pos >= 0.75:
        weight = -5
        score += weight
        add_factor(rows, "Range position", "PULLBACK RISK", weight, "Price is near the upper quarter of recent range.")
    else:
        add_factor(rows, "Range position", "NEUTRAL", 0, "Price is mid-range.")

    return score, rows


def calculate_whipsaw_probability(atr: float, data_quality: float, factor_score: float, news_score: int, history_adj: Dict[str, float]) -> float:
    """Whipsaw rises when signals conflict or data quality is poor."""
    base = 14.0

    # Weak net directional edge means more whipsaw risk.
    if abs(factor_score + news_score * 0.35) < 12:
        base += 6
    if data_quality < 75:
        base += 6
    if atr <= 0:
        base += 4

    base += float(history_adj.get("whipsaw", 0.0))
    return clamp(base, 8, 32)


def probability_engine(
    snapshot: Dict[str, Any],
    news_score: int,
    data_quality: float,
    history_adj: Dict[str, float],
) -> Dict[str, Any]:
    """Main deterministic probability engine.

    Output is intentionally bounded. This is a decision-support model, not a
    guarantee. The factor table provides the defense for every percentage.
    """
    technical_score, rows = technical_factor_score(snapshot)

    # Convert news score into directional model points.
    news_weighted = news_score * 0.45
    if news_score >= 25:
        add_factor(rows, "News pressure", "BULLISH", news_weighted, "Fresh headlines score as supply-risk bullish.")
    elif news_score <= -25:
        add_factor(rows, "News pressure", "BEARISH", news_weighted, "Fresh headlines score as de-escalation/demand bearish.")
    elif news_score != 0:
        add_factor(rows, "News pressure", "MILD", news_weighted, "Headlines have mild directional pressure.")
    else:
        add_factor(rows, "News pressure", "NEUTRAL", 0, "No strong headline pressure detected.")

    hist_up = float(history_adj.get("up", 0.0))
    hist_down = float(history_adj.get("down", 0.0))
    hist_net = hist_up - hist_down
    if abs(hist_net) >= 1:
        add_factor(rows, "Historical weekend cases", "ADJUSTMENT", hist_net, "Past weekend openings with similar bias adjust the model.")
    else:
        add_factor(rows, "Historical weekend cases", "LOW IMPACT", 0, "Not enough similar history to strongly adjust.")

    data_quality_penalty = 0.0
    if data_quality < 80:
        data_quality_penalty = -(80 - data_quality) * 0.15
        add_factor(rows, "Data quality", "PENALTY", data_quality_penalty, "Lower data quality reduces confidence and edge.")
    else:
        add_factor(rows, "Data quality", "OK", 0, "Data quality is acceptable.")

    total_edge = technical_score + news_weighted + hist_net + data_quality_penalty

    # Convert edge to probabilities. Keep outputs conservative.
    whipsaw = calculate_whipsaw_probability(
        atr=float(snapshot.get("atr", 0.0)),
        data_quality=data_quality,
        factor_score=technical_score,
        news_score=news_score,
        history_adj=history_adj,
    )

    directional_pool = 100.0 - whipsaw
    up_share = 1.0 / (1.0 + math.exp(-total_edge / 22.0))
    prob_up = directional_pool * up_share
    prob_down = directional_pool - prob_up

    # Conservative cap unless strong data quality.
    max_directional = 86 if data_quality >= 80 else 78
    prob_up = clamp(prob_up, 4, max_directional)
    prob_down = clamp(prob_down, 4, max_directional)

    # Renormalize.
    total = prob_up + prob_down + whipsaw
    prob_up = prob_up / total * 100
    prob_down = prob_down / total * 100
    whipsaw = whipsaw / total * 100

    direction = "SPIKE UP" if prob_up > prob_down and prob_up > whipsaw else "SPIKE DOWN" if prob_down > prob_up and prob_down > whipsaw else "WHIPSAW"
    confidence = max(prob_up, prob_down, whipsaw)

    if data_quality < CONFIG.min_data_quality:
        permission = "BLOCKED - DATA QUALITY"
    elif confidence < CONFIG.min_signal_confidence:
        permission = "BLOCKED - LOW EDGE"
    else:
        permission = "ALLOWED"

    risk_label = "LOW" if confidence >= 72 and data_quality >= 85 else "MEDIUM" if confidence >= 60 and data_quality >= 70 else "HIGH"

    # Suggested trade numbers are for planning only, not automatic execution.
    price = float(snapshot["price"])
    atr = max(float(snapshot.get("atr", 0.0)), 0.10)
    if direction == "SPIKE UP":
        stop = price - 1.25 * atr
        target = price + 2.0 * atr
    elif direction == "SPIKE DOWN":
        stop = price + 1.25 * atr
        target = price - 2.0 * atr
    else:
        stop = price
        target = price

    return {
        "prob_up": round(prob_up, 1),
        "prob_down": round(prob_down, 1),
        "prob_whipsaw": round(whipsaw, 1),
        "confidence": round(confidence, 1),
        "direction": direction,
        "permission": permission,
        "factor_score": round(total_edge, 2),
        "technical_score": round(technical_score, 2),
        "factor_rows": rows,
        "risk_label": risk_label,
        "entry": round(price, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
    }


# =============================================================================
# 10. AI EXPLANATION LAYER - NO DECISION POWER
# =============================================================================


def get_valid_gemini_model() -> str:
    """Return a Gemini model supporting generateContent.

    If the model listing fails, use a common fallback.
    """
    if not CONFIG.gemini_key:
        return ""

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={CONFIG.gemini_key}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if "models" in data:
            for model in data["models"]:
                name = model.get("name", "")
                methods = model.get("supportedGenerationMethods", [])
                if "generateContent" in methods and "flash" in name:
                    return name
            for model in data["models"]:
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    return model.get("name", "")
    except Exception:
        pass
    return "models/gemini-1.5-flash"


def generate_ai_explanation(
    snapshot: Dict[str, Any],
    probabilities: Dict[str, Any],
    scored_news: List[Dict[str, Any]],
    history_summary: str,
    chart_file: Optional[str],
) -> str:
    """Ask Gemini to explain, not decide.

    The prompt explicitly forbids changing probabilities.
    """
    if not CONFIG.gemini_key:
        return generate_fallback_explanation(snapshot, probabilities, scored_news, history_summary)

    model_name = get_valid_gemini_model()
    if not model_name:
        return generate_fallback_explanation(snapshot, probabilities, scored_news, history_summary)

    top_factors = probabilities.get("factor_rows", [])[:10]
    top_news = scored_news[:5]

    prompt = f"""
You are the explanation layer for a WTI crude oil opening-spike probability bot.

CRITICAL RULE:
You are NOT allowed to change the probabilities, direction, stop, or target.
You must only explain the deterministic model output below.

MODEL OUTPUT:
- Direction: {probabilities['direction']}
- Signal permission: {probabilities['permission']}
- Spike UP probability: {probabilities['prob_up']}%
- Spike DOWN probability: {probabilities['prob_down']}%
- Whipsaw/no-clean-spike probability: {probabilities['prob_whipsaw']}%
- Model confidence: {probabilities['confidence']}%
- Risk label: {probabilities['risk_label']}
- Entry reference: {probabilities['entry']}
- Stop reference: {probabilities['stop']}
- Target reference: {probabilities['target']}

MARKET SNAPSHOT:
- Price: {snapshot['price']:.2f}
- RSI: {snapshot['rsi']:.2f}
- ATR: {snapshot['atr']:.2f}
- EMA status: {snapshot['ema_status']}
- Trend: {snapshot['trend']}

HISTORICAL SUMMARY:
{history_summary}

FACTOR TABLE:
{json.dumps(top_factors, indent=2)}

TOP NEWS:
{json.dumps(top_news, indent=2)}

Write a concise defendable report with this exact structure:

🎯 MODEL DECISION
[1 sentence. Include probabilities exactly as given.]

🧠 WHY THIS IS THE OUTPUT
- Technical: [1 sentence]
- News: [1 sentence]
- Historical: [1 sentence]
- Risk: [1 sentence]

🛑 DEFENSE NOTE
[1 sentence explaining that this is probability-based decision support, not a guarantee.]
""".strip()

    parts: List[Dict[str, Any]] = [{"text": prompt}]
    if chart_file and os.path.exists(chart_file):
        try:
            with open(chart_file, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")
            parts.append({"inline_data": {"mime_type": "image/png", "data": encoded_image}})
        except Exception:
            pass

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={CONFIG.gemini_key}"
        resp = requests.post(
            url,
            json={"contents": [{"parts": parts}]},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return generate_fallback_explanation(snapshot, probabilities, scored_news, history_summary) + f"\n\nAI explanation unavailable: HTTP {resp.status_code}"
    except Exception as exc:
        return generate_fallback_explanation(snapshot, probabilities, scored_news, history_summary) + f"\n\nAI explanation unavailable: {exc}"


def generate_fallback_explanation(
    snapshot: Dict[str, Any],
    probabilities: Dict[str, Any],
    scored_news: List[Dict[str, Any]],
    history_summary: str,
) -> str:
    top_factor_rows = probabilities.get("factor_rows", [])[:4]
    factor_text = "; ".join([f"{r['name']} {r['signal']} ({r['weight']:+.1f})" for r in top_factor_rows])
    top_news = scored_news[0]["title"] if scored_news else "No strong fresh headline detected"

    return (
        f"🎯 MODEL DECISION\n"
        f"The model output is {probabilities['direction']} with UP {probabilities['prob_up']}%, "
        f"DOWN {probabilities['prob_down']}%, and WHIPSAW {probabilities['prob_whipsaw']}%.\n\n"
        f"🧠 WHY THIS IS THE OUTPUT\n"
        f"- Technical: {factor_text}.\n"
        f"- News: {top_news}.\n"
        f"- Historical: {history_summary}\n"
        f"- Risk: Permission is {probabilities['permission']} and risk is {probabilities['risk_label']}.\n\n"
        f"🛑 DEFENSE NOTE\n"
        f"This is a transparent probability model for the first opening spike only, not a guarantee of later trend direction."
    )


# =============================================================================
# 11. LOGGING AND CALIBRATION
# =============================================================================


def log_prediction(snapshot: Dict[str, Any], probabilities: Dict[str, Any], news_score: int, data_quality: float) -> None:
    ensure_data_files()
    path = CONFIG.data_dir / CONFIG.prediction_log_csv
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                utc_now_str(),
                CONFIG.ticker,
                round(float(snapshot["price"]), 4),
                probabilities["prob_up"],
                probabilities["prob_down"],
                probabilities["prob_whipsaw"],
                probabilities["confidence"],
                probabilities["direction"],
                probabilities["permission"],
                news_score,
                round(data_quality, 2),
                probabilities["factor_score"],
                probabilities["risk_label"],
                "",
                "actual_direction_later can be filled manually after reopen",
            ]
        )


def calibration_summary() -> str:
    """Summarize completed predictions where actual_direction_later is filled."""
    path = CONFIG.data_dir / CONFIG.prediction_log_csv
    if not path.exists():
        return "No prediction log yet."

    try:
        df = pd.read_csv(path)
    except Exception:
        return "Could not read prediction log."

    if df.empty or "actual_direction_later" not in df.columns:
        return "No calibration data yet."

    done = df.dropna(subset=["actual_direction_later"]).copy()
    done = done[done["actual_direction_later"].astype(str).str.strip() != ""]
    if done.empty:
        return "No completed calibration yet. Fill actual_direction_later after reopen."

    def predicted(row: pd.Series) -> str:
        probs = {
            "UP": float(row.get("prob_up", 0)),
            "DOWN": float(row.get("prob_down", 0)),
            "WHIPSAW": float(row.get("prob_whipsaw", 0)),
        }
        return max(probs, key=probs.get)

    done["predicted_class"] = done.apply(predicted, axis=1)
    done["actual_class"] = done["actual_direction_later"].astype(str).str.upper().str.strip()
    accuracy = (done["predicted_class"] == done["actual_class"]).mean() * 100
    return f"Completed calibration cases: {len(done)}. Direction-class accuracy: {accuracy:.1f}%."


# =============================================================================
# 12. TELEGRAM ALERTS
# =============================================================================


def telegram_configured() -> bool:
    return bool(CONFIG.telegram_token and CONFIG.telegram_chat_id)


def send_telegram_message(text: str) -> None:
    if not telegram_configured():
        return
    try:
        url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
        requests.post(url, data={"chat_id": CONFIG.telegram_chat_id, "text": text}, timeout=15)
    except Exception as exc:
        print(f"Telegram message error: {exc}")


def send_telegram_photo(chart_file: Optional[str]) -> None:
    if not telegram_configured() or not chart_file or not os.path.exists(chart_file):
        return
    try:
        url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendPhoto"
        with open(chart_file, "rb") as f:
            requests.post(url, data={"chat_id": CONFIG.telegram_chat_id}, files={"photo": f}, timeout=30)
    except Exception as exc:
        print(f"Telegram photo error: {exc}")


def send_model_alert(probabilities: Dict[str, Any], snapshot: Dict[str, Any], news_bias: str, reason: str, chart_file: Optional[str]) -> None:
    if not telegram_configured():
        return

    if CONFIG.mute_blocked_signals and probabilities["permission"] != "ALLOWED":
        return

    send_telegram_photo(chart_file)
    text = (
        f"🚨 WTI OPENING-SPIKE MODEL ALERT\n"
        f"Reason: {reason}\n"
        f"Time: {utc_now_str()}\n"
        f"Price: ${snapshot['price']:.2f}\n\n"
        f"Direction: {probabilities['direction']}\n"
        f"UP: {probabilities['prob_up']}% | DOWN: {probabilities['prob_down']}% | WHIPSAW: {probabilities['prob_whipsaw']}%\n"
        f"Confidence: {probabilities['confidence']}%\n"
        f"Permission: {probabilities['permission']}\n"
        f"Risk: {probabilities['risk_label']}\n"
        f"News bias: {news_bias}\n"
        f"Entry ref: {probabilities['entry']} | Stop ref: {probabilities['stop']} | Target ref: {probabilities['target']}\n\n"
        f"Note: First opening spike only, not later trend."
    )
    send_telegram_message(text)


# =============================================================================
# 13. DASHBOARD HTML
# =============================================================================


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WTI Opening-Spike Probability Bot v2</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap" rel="stylesheet">
    <style>
        body { background-color: #0b0f19; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
        .glass { background: rgba(15, 23, 42, 0.82); border: 1px solid #1e293b; border-radius: 12px; }
        .green { color: #4ade80; text-shadow: 0 0 10px rgba(74, 222, 128, 0.35); }
        .red { color: #f87171; text-shadow: 0 0 10px rgba(248, 113, 113, 0.35); }
        .yellow { color: #facc15; text-shadow: 0 0 10px rgba(250, 204, 21, 0.25); }
        .blue { color: #60a5fa; }
        .blink { animation: blinker 1.5s linear infinite; }
        @keyframes blinker { 50% { opacity: 0; } }
        .bar-bg { background: #1e293b; height: 10px; border-radius: 9999px; overflow: hidden; }
        .bar-fill { height: 10px; border-radius: 9999px; }
        pre { white-space: pre-wrap; word-break: break-word; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="max-w-7xl mx-auto space-y-6">
        <header class="flex flex-col md:flex-row md:items-end md:justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
                <h1 class="text-2xl md:text-3xl font-extrabold tracking-wide">WTI OPENING-SPIKE BOT <span class="text-xs green align-top blink">● LIVE</span></h1>
                <p class="text-slate-400 text-sm mt-1">First reopening spike/gap direction only — not later trend prediction.</p>
            </div>
            <div class="text-xs md:text-sm text-slate-400">
                Status: <span class="yellow">{{ data.status }}</span><br>
                Last update: {{ data.last_update }} | Ticker: {{ data.ticker }}
            </div>
        </header>

        <section class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div class="glass p-5">
                <p class="text-slate-400 text-xs uppercase">Current Price</p>
                <h2 class="text-5xl font-extrabold mt-2 {% if 'BULLISH' in data.trend %}green{% else %}red{% endif %}">${{ "%.2f"|format(data.price) }}</h2>
                <p class="text-xs text-slate-500 mt-2">{{ data.trend }} | {{ data.ema_status }}</p>
            </div>

            <div class="glass p-5 md:col-span-2">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <p class="text-slate-400 text-xs uppercase">Model Decision</p>
                        <h2 class="text-3xl font-extrabold mt-1 {% if data.signal_direction == 'SPIKE UP' %}green{% elif data.signal_direction == 'SPIKE DOWN' %}red{% else %}yellow{% endif %}">{{ data.signal_direction }}</h2>
                        <p class="text-xs text-slate-400 mt-1">Permission: <span class="yellow">{{ data.signal_permission }}</span></p>
                    </div>
                    <div class="text-right">
                        <p class="text-slate-400 text-xs uppercase">Confidence</p>
                        <p class="text-3xl font-bold yellow">{{ "%.1f"|format(data.model_confidence) }}%</p>
                        <p class="text-xs text-slate-500">Risk: {{ data.risk_label }}</p>
                    </div>
                </div>

                <div class="space-y-3">
                    <div>
                        <div class="flex justify-between text-xs mb-1"><span>Spike UP</span><span>{{ "%.1f"|format(data.prob_up) }}%</span></div>
                        <div class="bar-bg"><div class="bar-fill bg-green-500" style="width: {{ data.prob_up }}%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1"><span>Spike DOWN</span><span>{{ "%.1f"|format(data.prob_down) }}%</span></div>
                        <div class="bar-bg"><div class="bar-fill bg-red-500" style="width: {{ data.prob_down }}%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1"><span>Whipsaw / no clean spike</span><span>{{ "%.1f"|format(data.prob_whipsaw) }}%</span></div>
                        <div class="bar-bg"><div class="bar-fill bg-yellow-400" style="width: {{ data.prob_whipsaw }}%"></div></div>
                    </div>
                </div>
            </div>

            <div class="glass p-5">
                <p class="text-slate-400 text-xs uppercase border-b border-slate-700 pb-2">Data / Technicals</p>
                <div class="text-sm space-y-2 mt-3">
                    <div class="flex justify-between"><span>RSI</span><span>{{ "%.2f"|format(data.rsi) }}</span></div>
                    <div class="flex justify-between"><span>ATR</span><span>${{ "%.2f"|format(data.atr) }}</span></div>
                    <div class="flex justify-between"><span>Data quality</span><span class="yellow">{{ "%.0f"|format(data.data_quality) }}%</span></div>
                    <div class="flex justify-between"><span>News score</span><span class="{% if data.news_score > 0 %}green{% elif data.news_score < 0 %}red{% else %}text-slate-300{% endif %}">{{ data.news_score }}</span></div>
                    <div class="text-xs text-slate-400 pt-2">{{ data.news_bias }}</div>
                </div>
            </div>
        </section>

        <section class="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div class="glass p-5 lg:col-span-2">
                <p class="text-slate-400 text-xs uppercase mb-3 border-b border-slate-700 pb-2">Chart Evidence</p>
                <img src="/chart" alt="WTI chart" class="w-full rounded-lg border border-slate-700">
            </div>
            <div class="glass p-5">
                <p class="text-slate-400 text-xs uppercase mb-3 border-b border-slate-700 pb-2">Trade Reference Only</p>
                <div class="grid grid-cols-3 gap-3 text-center">
                    <div class="bg-slate-900 rounded-lg p-3">
                        <p class="text-xs text-slate-500">Entry</p>
                        <p class="text-lg font-bold">${{ "%.2f"|format(data.trade_entry) }}</p>
                    </div>
                    <div class="bg-slate-900 rounded-lg p-3">
                        <p class="text-xs text-slate-500">Stop</p>
                        <p class="text-lg font-bold red">${{ "%.2f"|format(data.trade_stop) }}</p>
                    </div>
                    <div class="bg-slate-900 rounded-lg p-3">
                        <p class="text-xs text-slate-500">Target</p>
                        <p class="text-lg font-bold green">${{ "%.2f"|format(data.trade_target) }}</p>
                    </div>
                </div>
                <div class="mt-4 text-xs text-slate-400 space-y-2">
                    <p>{{ data.historical_summary }}</p>
                    <p>{{ data.calibration_summary }}</p>
                    {% if data.data_warnings %}
                    <div class="mt-3 p-3 bg-red-950/40 border border-red-900 rounded-lg">
                        <p class="red font-bold mb-1">Data warnings</p>
                        {% for w in data.data_warnings %}<p>• {{ w }}</p>{% endfor %}
                    </div>
                    {% endif %}
                </div>
            </div>
        </section>

        <section class="glass p-5">
            <p class="text-slate-400 text-xs uppercase mb-3 border-b border-slate-700 pb-2">Defendable Factor Table</p>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead class="text-slate-400 text-xs uppercase">
                        <tr class="border-b border-slate-700">
                            <th class="text-left py-2">Factor</th>
                            <th class="text-left py-2">Signal</th>
                            <th class="text-right py-2">Weight</th>
                            <th class="text-left py-2 pl-4">Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in data.factor_rows %}
                        <tr class="border-b border-slate-800">
                            <td class="py-2">{{ row.name }}</td>
                            <td class="py-2 {% if row.weight > 0 %}green{% elif row.weight < 0 %}red{% else %}text-slate-300{% endif %}">{{ row.signal }}</td>
                            <td class="py-2 text-right {% if row.weight > 0 %}green{% elif row.weight < 0 %}red{% else %}text-slate-300{% endif %}">{{ "%+.2f"|format(row.weight) }}</td>
                            <td class="py-2 pl-4 text-slate-400">{{ row.reason }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </section>

        <section class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="glass p-5">
                <p class="text-slate-400 text-xs uppercase mb-3 border-b border-slate-700 pb-2">Explanation Layer</p>
                <pre class="text-slate-300 text-sm leading-relaxed">{{ data.explanation }}</pre>
            </div>
            <div class="glass p-5 max-h-[520px] overflow-y-auto">
                <p class="text-slate-400 text-xs uppercase mb-3 border-b border-slate-700 pb-2">48-Hour Scored Macro Timeline</p>
                <div class="space-y-4">
                    {% for item in data.news %}
                    <div class="border-l-2 {% if item.score > 0 %}border-green-500{% elif item.score < 0 %}border-red-500{% else %}border-slate-600{% endif %} pl-3">
                        <p class="text-xs text-slate-500">{{ item.time }}</p>
                        <a href="{{ item.link }}" target="_blank" class="text-sm text-slate-200 hover:text-white block">{{ item.title }}</a>
                        <p class="text-xs mt-1 {% if item.score > 0 %}green{% elif item.score < 0 %}red{% else %}text-slate-400{% endif %}">Score: {{ item.score }} | {{ item.reasons }}</p>
                    </div>
                    {% endfor %}
                    {% if not data.news %}<p class="text-slate-500 text-sm">No fresh headlines loaded.</p>{% endif %}
                </div>
            </div>
        </section>
    </div>
    <script>setTimeout(function(){ location.reload(); }, 60000);</script>
</body>
</html>
"""


@app.route("/")
def dashboard() -> str:
    return render_template_string(HTML_TEMPLATE, data=DASHBOARD_DATA)


@app.route("/chart")
def serve_chart():
    if os.path.exists(CONFIG.chart_file):
        return send_file(CONFIG.chart_file, mimetype="image/png")
    return "Chart not generated yet. Refresh after first data cycle.", 404


@app.route("/api/state")
def api_state():
    return DASHBOARD_DATA


# =============================================================================
# 14. MAIN BOT LOOP
# =============================================================================


def detect_breaking_news(raw_entries: List[Any], seen_links: set) -> Tuple[bool, str]:
    """Detect fresh unseen news within 15 minutes."""
    current_utc = calendar.timegm(time.gmtime())
    for entry in raw_entries:
        if "published_parsed" not in entry:
            continue
        age_seconds = current_utc - calendar.timegm(entry.published_parsed)
        if age_seconds < 900 and entry.link not in seen_links:
            seen_links.add(entry.link)
            return True, f"BREAKING: {entry.title.split(' - ')[0]}"
    return False, "Regular scheduled report"


def run_analysis_cycle(reason: str = "Manual/initial cycle") -> None:
    """Run one full analysis cycle and update dashboard."""
    global DASHBOARD_DATA

    DASHBOARD_DATA["status"] = "Fetching market data"
    intraday, hourly = download_market_data()

    # Auto-extract weekend history from available 15m data.
    try:
        cases = extract_weekend_cases_from_intraday(intraday)
        upsert_weekend_history(cases)
    except Exception as exc:
        print(f"Weekend history extraction warning: {exc}")

    snapshot = latest_market_snapshot(intraday, hourly)
    data_quality, data_warnings = assess_data_quality(intraday, hourly, snapshot)

    DASHBOARD_DATA["status"] = "Fetching and scoring news"
    headlines, raw_entries = fetch_news()
    news_score, news_bias, scored_news = score_news(headlines)

    history = load_weekend_history()
    history_summary, history_adj = summarize_history(history, news_score)

    DASHBOARD_DATA["status"] = "Calculating deterministic probabilities"
    probabilities = probability_engine(snapshot, news_score, data_quality, history_adj)

    chart_data = hourly.tail(250) if not hourly.empty else intraday.tail(250)
    chart_file = create_chart(chart_data, snapshot)

    DASHBOARD_DATA["status"] = "Generating explanation"
    explanation = generate_ai_explanation(snapshot, probabilities, scored_news, history_summary, chart_file)

    # Audit log every model report.
    try:
        log_prediction(snapshot, probabilities, news_score, data_quality)
    except Exception as exc:
        print(f"Prediction log warning: {exc}")

    cal_summary = calibration_summary()

    DASHBOARD_DATA.update(
        {
            "status": "Monitoring active",
            "last_update": utc_now_str(),
            "ticker": CONFIG.ticker,
            "price": float(snapshot["price"]),
            "trend": snapshot["trend"],
            "rsi": float(snapshot["rsi"]),
            "atr": float(snapshot["atr"]),
            "ema_status": snapshot["ema_status"],
            "data_quality": data_quality,
            "data_warnings": data_warnings,
            "news": scored_news,
            "news_score": news_score,
            "news_bias": news_bias,
            "prob_up": probabilities["prob_up"],
            "prob_down": probabilities["prob_down"],
            "prob_whipsaw": probabilities["prob_whipsaw"],
            "model_confidence": probabilities["confidence"],
            "signal_permission": probabilities["permission"],
            "signal_direction": probabilities["direction"],
            "factor_rows": probabilities["factor_rows"],
            "historical_summary": history_summary,
            "calibration_summary": cal_summary,
            "explanation": explanation,
            "trade_entry": probabilities["entry"],
            "trade_stop": probabilities["stop"],
            "trade_target": probabilities["target"],
            "risk_label": probabilities["risk_label"],
        }
    )

    send_model_alert(probabilities, snapshot, news_bias, reason, chart_file)


def run_bot_loop() -> None:
    """Continuous monitoring loop."""
    ensure_data_files()
    last_report_time = 0.0
    last_price = 0.0
    seen_news_links: set = set()

    while True:
        try:
            # Lightweight check includes news and price, but full report runs by time or break.
            intraday, hourly = download_market_data()
            snapshot = latest_market_snapshot(intraday, hourly)
            price = float(snapshot["price"])
            headlines, raw_entries = fetch_news()
            is_breaking, breaking_reason = detect_breaking_news(raw_entries, seen_news_links)

            # Fast price spike alert, separate from full model.
            if last_price > 0:
                diff = price - last_price
                atr = max(float(snapshot.get("atr", 0.0)), 0.1)
                if abs(diff) >= max(0.50, atr * 0.60):
                    direction = "UP 🟢" if diff > 0 else "DOWN 🔴"
                    send_telegram_message(
                        f"⚡ WTI FAST MOVE ALERT\nMoved ${abs(diff):.2f} {direction}\nPrice: ${price:.2f}\nTime: {utc_now_str()}"
                    )

            current_time = time.time()
            is_time_up = (current_time - last_report_time) >= CONFIG.report_interval_seconds

            if is_breaking or is_time_up:
                reason = breaking_reason if is_breaking else "Scheduled model refresh"
                run_analysis_cycle(reason=reason)
                last_report_time = time.time()

            last_price = price
            time.sleep(CONFIG.loop_sleep_seconds)

        except Exception as exc:
            DASHBOARD_DATA["status"] = f"Error: {exc}"
            DASHBOARD_DATA["last_update"] = utc_now_str()
            print(f"Bot loop error: {exc}")
            time.sleep(CONFIG.loop_sleep_seconds)


# =============================================================================
# 15. ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    ensure_data_files()
    print("🚀 Starting WTI Opening-Spike Probability Bot v2")
    print(f"Ticker: {CONFIG.ticker}")
    print(f"Dashboard: http://0.0.0.0:{CONFIG.port}")
    print("Decision engine: deterministic probability model")
    print("AI role: explanation only")

    # Initial cycle before web traffic arrives. If it fails, the loop will retry.
    try:
        run_analysis_cycle(reason="Initial startup model run")
    except Exception as exc:
        print(f"Initial run warning: {exc}")
        DASHBOARD_DATA["status"] = f"Initial run failed: {exc}"
        DASHBOARD_DATA["last_update"] = utc_now_str()

    bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
    bot_thread.start()

    app.run(host="0.0.0.0", port=CONFIG.port)
