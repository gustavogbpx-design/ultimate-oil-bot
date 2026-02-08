import os
import time
import yfinance as yf
import requests
import feedparser
import pandas as pd
import mplfinance as mpf
from ta.momentum import RSIIndicator
from ta.trend import MACD

# --- CONFIG ---
# This makes sure the bot doesn't crash if keys are missing
try:
    GEMINI_KEY = os.environ["GEMINI_API_KEY"]
    TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
    TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
except KeyError:
    print("CRITICAL ERROR: API Keys are missing in Railway Variables!")
    exit(1)

# --- FUNCTIONS ---
def get_market_data():
    ticker = "CL=F"
    try:
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, "No Data"
        
        # Fix MultiIndex columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        # heavy lifting to fix pandas formatting
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1:
            close = close.iloc[:, 0]
        data["Close"] = close

        # Indicators
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        
        price = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        
        # Simple Trend Logic
        if data["MACD"].iloc[-1] > data["Signal"].iloc[-1]:
            trend = "BULLISH 🟢"
        else:
            trend = "BEARISH 🔴"
        
        return data, price, rsi, trend
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, 0, 0, "Error"

def create_chart(data):
    if data is None: return None
    fname = "oil_chart.png"
    mpf.plot(data.tail(40), type='candle', style='charles', volume=False, savefig=fname)
    return fname

def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Iran+Conflict&hl=en-US&gl=US&ceid=US:en")
        return [entry.title for entry in feed.entries[:5]]
    except:
        return []

def analyze_with_gemini(price, rsi, trend, headlines):
    news_text = "\n".join([f"- {h}" for h in headlines])
    prompt = f"""
    Act as a Hedge Fund Algo.
    
    MARKET DATA:
    - WTI Oil Price: ${price:.2f}
    - RSI: {rsi:.2f}
    - Trend: {trend}
    
    NEWS:
    {news_text}
    
    TASK:
    Analyze the setup. Is this a BUY, SELL, or WAIT?
    Keep it short (under 100 words).
    """
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
    try:
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        if resp.status_code == 200:
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Gemini Error: {e}")
    return "⚠️ AI Analysis Unavailable."

def send_telegram(price, trend, analysis, chart_file):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    
    # Send Chart
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})

    # Send Text
    text = f"🛢 **WTI OIL ALERT**\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- THE MAGICAL LOOP (RUNS FOREVER) ---
if __name__ == "__main__":
    print("🚀 Bot starting 24/7 mode on Railway...")
    
    while True:
        try:
            print("Checking market...")
            data, price, rsi, trend = get_market_data()
            
            if data is not None:
                chart = create_chart(data)
                headlines = get_news()
                analysis = analyze_with_gemini(price, rsi, trend, headlines)
                send_telegram(price, trend, analysis, chart)
                print("✅ Update sent!")
            else:
                print("❌ No data found.")

        except Exception as e:
            print(f"⚠️ Crash prevented: {e}")
        
        # SLEEP FOR 15 MINUTES (900 seconds)
        print("💤 Sleeping for 15 minutes...")
        time.sleep(900)
