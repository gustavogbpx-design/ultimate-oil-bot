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
from datetime import datetime

# --- 1. SETUP KEYS ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- 2. GET DATA (SPEED MODE) ---
def get_market_data():
    ticker = "CL=F"
    try:
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, "No Data", 0, 0, 0
        
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
        
        # SPEED EMAs (21 & 50)
        data["EMA50"] = EMAIndicator(close=data["Close"], window=50).ema_indicator()
        data["EMA21"] = EMAIndicator(close=data["Close"], window=21).ema_indicator()
        
        price = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        atr = data["ATR"].iloc[-1]
        ema50 = data["EMA50"].iloc[-1]
        ema21 = data["EMA21"].iloc[-1]
        
        trend = "BULLISH 🟢" if data["MACD"].iloc[-1] > data["Signal"].iloc[-1] else "BEARISH 🔴"
        
        return data, price, rsi, trend, atr, ema50, ema21
        
    except Exception as e:
        print(f"Data Error: {e}")
        return None, 0, 0, "Error", 0, 0, 0

# --- 3. DRAW CHART ---
def create_chart(data):
    if data is None: return None
    fname = "oil_chart.png"
    mpf.plot(data.tail(50), type='candle', style='charles', volume=False, mav=(21, 50), savefig=fname)
    return fname

# --- 4. GET NEWS (WITH TIMESTAMPS) ---
def get_news():
    try:
        # Sort by "Date" to get the absolute newest stuff
        url = "https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Inventory+OR+Iran&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        if not feed.entries: return []
        
        # Format: "Headline (Published: Time)"
        headlines = []
        for entry in feed.entries[:5]:
            pub_time = entry.get('published', 'Unknown Time')
            headlines.append(f"{entry.title} (Time: {pub_time})")
            
        return headlines
    except:
        return ["⚠️ News Feed Unavailable"]

# --- 5. FIND MODEL ---
def get_valid_model():
    # Try Flash first (Fastest), then others
    models = ["models/gemini-1.5-flash", "models/gemini-pro", "models/gemini-1.5-pro-latest"]
    for m in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{m}?key={GEMINI_KEY}"
            if requests.get(url).status_code == 200: return m
        except: continue
    return "models/gemini-1.5-flash"

# --- 6. ANALYZE (NEWS FIRST LOGIC) ---
def analyze_market(price, rsi, trend, atr, ema50, ema21, headlines):
    model_name = get_valid_model()
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    stop_loss_buy = price - (1.5 * atr)
    take_profit_buy = price + (2.5 * atr)
    stop_loss_sell = price + (1.5 * atr) 
    take_profit_sell = price - (2.5 * atr)
    
    ema_status = "BULLISH (21 > 50)" if ema21 > ema50 else "BEARISH (21 < 50)"

    # --- THE "NEWS FIRST" PROMPT ---
    prompt = f"""
    Act as a Hedge Fund Algo trading WTI Crude Oil.
    
    LATEST NEWS (CRITICAL):
    {news_text}
    
    TECHNICAL DATA (30-min Chart):
    - Price: ${price:.2f}
    - EMA Trend: {ema_status}
    - RSI: {rsi:.2f}
    - Volatility (ATR): {atr:.2f}
    
    TASK:
    1. ANALYZE NEWS SENTIMENT FIRST. Is the news Bullish (War/Supply Cuts) or Bearish (Inventory Build/Peace)?
    2. CROSS-REFERENCE with Technicals. 
       - If News says SELL but Chart says BUY -> ISSUE "WAIT" or "CAUTIOUS BUY".
       - If News & Chart agree -> ISSUE STRONG SIGNAL.
    3. Use the calculated limits below.
    
    CALCULATED LIMITS:
    - BUY Setup: Stop=${stop_loss_buy:.2f}, Target=${take_profit_buy:.2f}
    - SELL Setup: Stop=${stop_loss_sell:.2f}, Target=${take_profit_sell:.2f}
    
    OUTPUT FORMAT (Strictly follow this):
    
    📰 **NEWS SENTIMENT**
    [Bullish/Bearish/Neutral] because... [Explain in 1 sentence]
    
    💎 **TRADE DECISION**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛡️ Stop: [ATR Value]
    🎯 Target: [ATR Value]
    
    📊 **REASONING**
    - [Technical Analysis]
    - [News Impact Analysis]
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
    
    text = f"🌍 **WTI NEWS+TECH REPORT**\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("🚀 Bot Started (News-First Mode)...")
    while True:
        try:
            print("Analyzing market...")
            data, price, rsi, trend, atr, ema50, ema21 = get_market_data()
            if data is not None:
                chart = create_chart(data)
                headlines = get_news()
                analysis = analyze_market(price, rsi, trend, atr, ema50, ema21, headlines)
                send_telegram(price, trend, analysis, chart)
                print("✅ Report Sent!")
            else:
                print("❌ No data.")
        except Exception as e:
            print(f"⚠️ Error: {e}")
        print("💤 Sleeping for 10 minutes...")
        time.sleep(1800)
