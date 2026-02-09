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

# --- 2. GET DATA (UPDATED WITH ATR & EMA) ---
def get_market_data():
    ticker = "CL=F"
    try:
        # Download 5 Days of 30-min candles
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, "No Data", 0, 0
        
        # Clean Data Format
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close

        # --- NEW INDICATORS ---
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        
        # 1. ATR for Smart Stops (Volatility)
        data["ATR"] = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14).average_true_range()
        
        # 2. 200 EMA for Trend Filter
        data["EMA200"] = EMAIndicator(close=data["Close"], window=200).ema_indicator()
        
        price = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        atr = data["ATR"].iloc[-1]
        ema200 = data["EMA200"].iloc[-1]
        
        trend = "BULLISH 🟢" if data["MACD"].iloc[-1] > data["Signal"].iloc[-1] else "BEARISH 🔴"
        
        return data, price, rsi, trend, atr, ema200
        
    except Exception as e:
        print(f"Data Error: {e}")
        return None, 0, 0, "Error", 0, 0

# --- 3. DRAW CHART ---
def create_chart(data):
    if data is None: return None
    fname = "oil_chart.png"
    # Added EMA lines to the chart visual
    mpf.plot(data.tail(50), type='candle', style='charles', volume=False, 
             mav=(50, 200), savefig=fname)
    return fname

# --- 4. GET NEWS ---
def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Iran+Conflict&hl=en-US&gl=US&ceid=US:en")
        if not feed.entries: return []
        return [entry.title for entry in feed.entries[:5]]
    except:
        return []

# --- 5. FIND MODEL (UNIVERSAL FIX) ---
def get_valid_model():
    # Loop through models to prevent 404/429 errors
    models = ["models/gemini-1.5-flash", "models/gemini-pro", "models/gemini-1.5-pro-latest"]
    for m in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/{m}?key={GEMINI_KEY}"
            if requests.get(url).status_code == 200:
                return m
        except:
            continue
    return "models/gemini-1.5-flash"

# --- 6. ANALYZE (SMART DEFENSE MODE) ---
def analyze_market(price, rsi, trend, atr, ema200, headlines):
    model_name = get_valid_model()
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    # --- MATH: Calculate Smart Stops ---
    stop_loss_buy = price - (2.0 * atr)
    take_profit_buy = price + (3.0 * atr)
    stop_loss_sell = price + (2.0 * atr) 
    take_profit_sell = price - (3.0 * atr)
    
    ema_status = "Price is ABOVE 200 EMA (Uptrend)" if price > ema200 else "Price is BELOW 200 EMA (Downtrend)"

    prompt = f"""
    Act as a Senior Wall Street Trader (Risk Manager).
    
    MARKET DATA:
    - Price: ${price:.2f}
    - RSI: {rsi:.2f}
    - Trend: {trend}
    - Context: {ema_status}
    - Volatility (ATR): {atr:.2f}
    
    SMART STOPS (Calculated from ATR):
    - If BUY: Stop=${stop_loss_buy:.2f}, Target=${take_profit_buy:.2f}
    - If SELL: Stop=${stop_loss_sell:.2f}, Target=${take_profit_sell:.2f}
    
    NEWS:
    {news_text}
    
    TASK:
    1. Determine the best trade setup.
    2. USE THE CALCULATED STOPS above for risk management.
    3. Explain WHY based on the EMA context and News.
    
    OUTPUT FORMAT (Strictly follow this):
    
    💎 **TRADE SETUP**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛡️ Smart Stop: [Use ATR Value]
    🎯 Smart Target: [Use ATR Value]
    
    📊 **DEEP ANALYSIS**
    Risk Level: [Low/Med/High]
    Reasoning:
    - [Technicals: RSI + EMA]
    - [News Impact]
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
    
    text = f"🛢 **WTI MASTER REPORT (Phase 1)**\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP (RUNS FOREVER) ---
if __name__ == "__main__":
    print("🚀 Bot Started in 24/7 Phase 1 Mode...")
    while True:
        try:
            print("Analyzing market...")
            # Unpack new variables (atr, ema200)
            data, price, rsi, trend, atr, ema200 = get_market_data()
            
            if data is not None:
                chart = create_chart(data)
                headlines = get_news()
                # Pass new variables to analysis
                analysis = analyze_market(price, rsi, trend, atr, ema200, headlines)
                send_telegram(price, trend, analysis, chart)
                print("✅ Report Sent!")
            else:
                print("❌ No data received.")
        except Exception as e:
            print(f"⚠️ Crash prevention: {e}")
        
        print("💤 Sleeping for 30 minutes...")
        time.sleep(1800)
