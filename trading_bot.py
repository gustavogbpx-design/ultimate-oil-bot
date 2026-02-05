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
        
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close

        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        
        price = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        trend = "BULLISH 🟢" if data["MACD"].iloc[-1] > data["Signal"].iloc[-1] else "BEARISH 🔴"
        
        return data, price, rsi, trend
    except Exception as e:
        return None, 0, 0, f"Data Error: {str(e)}"

# --- 3. DRAW CHART ---
def create_chart(data):
    if data is None: return None
    fname = "oil_chart.png"
    mpf.plot(data.tail(40), type='candle', style='charles', volume=False, savefig=fname)
    return fname

# --- 4. GET NEWS ---
def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Iran+Conflict&hl=en-US&gl=US&ceid=US:en")
        if not feed.entries: return []
        return [entry.title for entry in feed.entries[:5]]
    except:
        return []

# --- 5. ANALYZE (DEBUG MODE) ---
def analyze_market(price, rsi, trend, headlines):
    debug_log = ""
    
    # STEP A: Try Google AI (Standard Flash Model)
    try:
        news_text = "\n".join([f"- {h}" for h in headlines])
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
        prompt = f"Market: Price ${price}, Trend {trend}. News: {news_text}. Buy or Sell? Short answer."
        
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        
        if resp.status_code == 200:
            return "🧠 **AI BRAIN ANALYSIS:**\n" + resp.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            # THIS IS THE FIX: IT WILL PRINT THE GOOGLE ERROR
            debug_log = f"⚠️ Google Error ({resp.status_code}): {resp.text}"
            
    except Exception as e:
        debug_log = f"⚠️ Connection Error: {str(e)}"
    
    # STEP B: Fallback Algo (If AI failed, we use this)
    score = 0
    matches = []
    keywords = {"WAR": 1, "CONFLICT": 1, "ATTACK": 1, "PEACE": -1, "TALKS": -1, "DEAL": -1}
    
    for h in headlines:
        for word, val in keywords.items():
            if word in h.upper():
                score += val
                matches.append(f"{word} ({val})")
                
    signal = "BUY 🟢" if score > 0 else "SELL 🔴" if score < 0 else "WAIT ✋"
    
    return f"{debug_log}\n\n⚙️ **BACKUP ALGO:**\nSignal: {signal}\nScore: {score}\nKeywords: {', '.join(matches)}"

# --- 6. SEND TELEGRAM ---
def send_telegram(price, rsi, trend, analysis, chart_file):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    
    text = f"🛢 **WTI DEBUG REPORT**\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN ---
if __name__ == "__main__":
    data, price, rsi, trend = get_market_data()
    if data is not None:
        chart = create_chart(data)
        headlines = get_news()
        analysis = analyze_market(price, rsi, trend, headlines)
        send_telegram(price, rsi, trend, analysis, chart)
