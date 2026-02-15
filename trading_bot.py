import os
import time
import yfinance as yf
import requests
import feedparser
import pandas as pd
import mplfinance as mpf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# --- 1. SETUP KEYS ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- 2. GET DATA (SPEED MODE + S/R LEVELS) ---
def get_market_data():
    ticker = "CL=F"
    try:
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, "No Data", 0, 0, 0, 0, 0
        
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close

        # INDICATORS
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        data["ATR"] = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14).average_true_range()
        
        # SPEED EMAs
        data["EMA50"] = EMAIndicator(close=data["Close"], window=50).ema_indicator()
        data["EMA21"] = EMAIndicator(close=data["Close"], window=21).ema_indicator()
        
        # --- NEW: CALCULATE SUPPORT & RESISTANCE (Last 50 Candles) ---
        recent_high = data["High"].tail(50).max()
        recent_low = data["Low"].tail(50).min()
        
        price = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        atr = data["ATR"].iloc[-1]
        ema50 = data["EMA50"].iloc[-1]
        ema21 = data["EMA21"].iloc[-1]
        
        trend = "BULLISH 🟢" if data["MACD"].iloc[-1] > data["Signal"].iloc[-1] else "BEARISH 🔴"
        
        # Returning 9 values now (Added Support & Resistance)
        return data, price, rsi, trend, atr, ema50, ema21, recent_high, recent_low
        
    except Exception as e:
        print(f"Data Error: {e}")
        return None, 0, 0, "Error", 0, 0, 0, 0, 0

# --- 3. DRAW CHART (WITH S/R LINES) ---
def create_chart(data, high_level, low_level):
    if data is None: return None
    fname = "oil_chart.png"
    
    # Add Horizontal Lines for Support (Green) and Resistance (Red)
    # hlines = dict(hlines=[high_level, low_level], colors=['red', 'green'], linewidths=[1, 1], alpha=0.5)
    
    # Draw Chart with EMAs
    mpf.plot(data.tail(50), type='candle', style='charles', volume=False, 
             mav=(21, 50), 
             hlines=dict(hlines=[high_level, low_level], colors=['red', 'green'], linestyle='-.'),
             savefig=fname)
    return fname

# --- 4. GET NEWS ---
def get_news():
    try:
        url = "https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Inventory+OR+Iran&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        if not feed.entries: return []
        headlines = []
        for entry in feed.entries[:5]:
            pub_time = entry.get('published', '')
            headlines.append(f"{entry.title} ({pub_time})")
        return headlines
    except:
        return ["⚠️ News Feed Unavailable"]

# --- 5. FIND MODEL (UNCHANGED ENGINE) ---
def get_valid_model():
    models = ["models/gemini-1.5-flash", "models/gemini-pro", "models/gemini-1.5-pro-latest"]
    for m in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{m}?key={GEMINI_KEY}"
            if requests.get(url).status_code == 200: return m
        except: continue
    return "models/gemini-1.5-flash"

# --- 6. ANALYZE (NOW SEES SUPPORT/RESISTANCE) ---
def analyze_market(price, rsi, trend, atr, ema50, ema21, res_level, sup_level, headlines):
    model_name = get_valid_model()
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    stop_loss_buy = price - (1.5 * atr)
    take_profit_buy = price + (2.5 * atr)
    stop_loss_sell = price + (1.5 * atr) 
    take_profit_sell = price - (2.5 * atr)
    
    ema_status = "BULLISH (21 > 50)" if ema21 > ema50 else "BEARISH (21 < 50)"

    # --- PROMPT WITH VISION ---
    prompt = f"""
    Act as a Professional Day Trader.
    
    LATEST NEWS:
    {news_text}
    
    TECHNICAL DATA (30-min Chart):
    - Price: ${price:.2f}
    - EMA Trend: {ema_status}
    - RSI: {rsi:.2f}
    - Volatility (ATR): {atr:.2f}
    
    KEY LEVELS (Auto-Detected):
    - 🔴 Resistance (Ceiling): ${res_level:.2f}
    - 🟢 Support (Floor): ${sup_level:.2f}
    
    TASK:
    1. Check if price is too close to Resistance (Don't Buy) or Support (Don't Sell).
    2. ANALYZE NEWS SENTIMENT.
    3. Issue a trade signal based on the Breakout or Rejection of these levels.
    
    CALCULATED LIMITS:
    - BUY Setup: Stop=${stop_loss_buy:.2f}, Target=${take_profit_buy:.2f}
    - SELL Setup: Stop=${stop_loss_sell:.2f}, Target=${take_profit_sell:.2f}
    
    OUTPUT FORMAT (Strictly follow this):
    
    🧱 **KEY LEVELS**
    Resistance: ${res_level:.2f} | Support: ${sup_level:.2f}
    
    💎 **TRADE DECISION**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛡️ Stop: [ATR Value]
    🎯 Target: [ATR Value]
    
    📊 **REASONING**
    - [News Sentiment]
    - [Distance to Key Levels]
    """

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_KEY}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        if resp.status_code == 200:
            ai_reply = resp.json()['candidates'][0]['content']['parts'][0]['text']
            return f"🧠 **AI SIGNAL ({model_name.split('/')[-1]}):**\n{ai_reply}"
        else:
            return f"⚠️ AI Error: {resp.status_code} - {resp.text}"
    except Exception as e:
        return f"⚠️ AI Connection Failed: {e}"

# --- 7. SEND TELEGRAM ---
def send_telegram(price, trend, analysis, chart_file):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    
    text = f"🛡️ **WTI VISION REPORT**\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("🚀 Bot Started (Vision Mode: S/R Lines)...")
    while True:
        try:
            print("Analyzing market...")
            # Unpack 9 values now (S/R included)
            data, price, rsi, trend, atr, ema50, ema21, res, sup = get_market_data()
            
            if data is not None:
                # Pass S/R levels to Chart and AI
                chart = create_chart(data, res, sup)
                headlines = get_news()
                analysis = analyze_market(price, rsi, trend, atr, ema50, ema21, res, sup, headlines)
                send_telegram(price, trend, analysis, chart)
                print("✅ Report Sent!")
            else:
                print("❌ No data.")
        except Exception as e:
            print(f"⚠️ Error: {e}")
        
        print("💤 Sleeping for 10 minutes...")
        time.sleep(1800)
