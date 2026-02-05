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

# --- 5. FIND MODEL (THE LOGIC THAT WORKED) ---
def get_valid_model():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}"
    try:
        resp = requests.get(url)
        data = resp.json()
        if 'models' in data:
            for model in data['models']:
                if 'generateContent' in model.get('supportedGenerationMethods', []):
                    return model['name']
    except:
        pass
    return "models/gemini-1.5-flash" # Fallback

# --- 6. ANALYZE (DETAILED HEDGE FUND MODE) ---
def analyze_market(price, rsi, trend, headlines):
    
    # 1. Get the working model name
    model_name = get_valid_model()
    
    # 2. Prepare the Smart Prompt
    news_text = "\n".join([f"- {h}" for h in headlines])
    prompt = f"""
    Act as a Wall Street Oil Trader.
    
    MARKET DATA:
    - Price: ${price:.2f}
    - RSI: {rsi:.2f} (30=Oversold, 70=Overbought)
    - Trend: {trend}
    
    NEWS HEADLINES:
    {news_text}
    
    TASK:
    Analyze the data and news. Provide a detailed trading signal.
    
    OUTPUT FORMAT:
    ACTION: [BUY / SELL / WAIT]
    RISK: [LOW / HIGH]
    
    REASONING:
    - [Bullet point 1]
    - [Bullet point 2]
    - [Bullet point 3]
    """

    # 3. Ask AI
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_KEY}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        if resp.status_code == 200:
            ai_reply = resp.json()['candidates'][0]['content']['parts'][0]['text']
            # Clean up formatting
            return f"🧠 **AI ANALYSIS ({model_name.split('/')[-1]}):**\n{ai_reply}"
    except:
        pass

    # 4. Backup Algo (Just in case)
    return "⚠️ AI Silent. Using Math:\n" + ("BUY 🟢" if rsi < 30 else "SELL 🔴" if rsi > 70 else "WAIT ✋")

# --- 7. SEND TELEGRAM ---
def send_telegram(price, rsi, trend, analysis, chart_file):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    
    text = f"🛢 **WTI REPORT**\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN ---
if __name__ == "__main__":
    data, price, rsi, trend = get_market_data()
    if data is not None:
        chart = create_chart(data)
        headlines = get_news()
        analysis = analyze_market(price, rsi, trend, headlines)
        send_telegram(price, rsi, trend, analysis, chart)
