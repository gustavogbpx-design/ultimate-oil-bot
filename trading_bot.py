import os
import yfinance as yf
import requests
import feedparser
import pandas as pd
import mplfinance as mpf
from ta.momentum import RSIIndicator
from ta.trend import MACD

# --- 1. SETUP KEYS ---
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# --- 2. GET DATA ---
def get_market_data():
    ticker = "CL=F"
    try:
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, "No Data"
        
        # Cleanup Data Structure
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close

        # Indicators
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        
        # Latest
        price = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        trend = "BULLISH 🟢" if data["MACD"].iloc[-1] > data["Signal"].iloc[-1] else "BEARISH 🔴"
        
        return data, price, rsi, trend
    except Exception as e:
        print(f"Data Error: {e}")
        return None, 0, 0, "Error"

# --- 3. DRAW CHART ---
def create_chart(data):
    if data is None: return None
    fname = "oil_chart.png"
    mpf.plot(data.tail(40), type='candle', style='charles', volume=False, savefig=fname)
    return fname

# --- 4. GET NEWS ---
def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC&hl=en-US&gl=US&ceid=US:en")
        return "\n".join([f"- {e.title}" for e in feed.entries[:3]])
    except:
        return "News unavailable."

# --- 5. ASK AI (MODEL HUNTER) ---
def ask_gemini(price, rsi, trend, news):
    # We will try these models in order. The first one that works wins.
    candidate_models = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
        "gemini-1.0-pro"
    ]
    
    prompt = f"Price: ${price:.2f}, RSI: {rsi:.2f}, Trend: {trend}. News: {news}. Buy or Sell WTI? Short answer."
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"
        try:
            resp = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            if resp.status_code == 200:
                return f"✅ ({model}): " + resp.json()['candidates'][0]['content']['parts'][0]['text']
        except:
            continue # Try the next model
            
    return "⚠️ All AI models failed. Please check Google AI Studio Terms of Service."

# --- 6. SEND TELEGRAM ---
def send_telegram(price, rsi, trend, ai_msg, chart_file):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    
    # 1. Send Chart
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})

    # 2. Send Text
    text = f"🛢 OIL UPDATE\nPrice: ${price:.2f}\nRSI: {rsi:.2f}\nTrend: {trend}\n\n🤖 AI SAYS:\n{ai_msg}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN ---
if __name__ == "__main__":
    print("Running...")
    data, price, rsi, trend = get_market_data()
    
    if data is not None:
        chart = create_chart(data)
        news = get_news()
        ai = ask_gemini(price, rsi, trend, news)
        send_telegram(price, rsi, trend, ai, chart)
        print("Sent successfully.")
    else:
        print("Failed to get data.")
