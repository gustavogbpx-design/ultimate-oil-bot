import os
import time
import yfinance as yf
import requests
import feedparser
import pandas as pd
import mplfinance as mpf
from ta.momentum import RSIIndicator
from ta.trend import MACD

# --- 1. SETUP KEYS ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- 2. GLOBAL SETTINGS ---
price_history = []  # To track price changes over 1 hour

# --- 3. GET DATA ---
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
        print(f"Data Error: {e}")
        return None, 0, 0, "Error"

# --- 4. DRAW CHART ---
def create_chart(data):
    if data is None: return None
    fname = "oil_chart.png"
    mpf.plot(data.tail(40), type='candle', style='charles', volume=False, savefig=fname)
    return fname

# --- 5. GET NEWS ---
def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Iran+Conflict&hl=en-US&gl=US&ceid=US:en")
        if not feed.entries: return []
        return [entry.title for entry in feed.entries[:5]]
    except:
        return []

# --- 6. FIND MODEL ---
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
    return "models/gemini-1.5-flash"

# --- 7. ANALYZE (RICH AI MODE) ---
def analyze_market(price, rsi, trend, headlines):
    model_name = get_valid_model()
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    prompt = f"""
    Act as a Senior Wall Street Trader.
    
    MARKET DATA:
    - Price: ${price:.2f}
    - RSI: {rsi:.2f}
    - Trend: {trend}
    
    NEWS:
    {news_text}
    
    TASK:
    Analyze the setup.
    
    OUTPUT FORMAT (Strictly follow this):
    
    💎 **TRADE SETUP**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛑 Stop Loss: [Price]
    🎯 Take Profit: [Price]
    
    📊 **DEEP ANALYSIS**
    Risk Level: [Low / Medium / High]
    Reasoning:
    - [Technicals]
    - [News]
    """

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_KEY}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        if resp.status_code == 200:
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        pass
    return "Analysis Failed"

# --- 8. SEND TELEGRAM ---
def send_telegram(price, trend, analysis, chart_file, alert_reason):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    
    text = f"🚨 **WTI ALERT ({alert_reason})**\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP (RUNS FOREVER) ---
if __name__ == "__main__":
    print("🚀 Bot Started (Filter Mode: Low/Med Risk OR $1 Move)...")
    
    while True:
        try:
            print("Checking market...")
            data, price, rsi, trend = get_market_data()
            
            if data is not None:
                # 1. Update Price History (Keep last 1 hour)
                current_time = time.time()
                price_history.append((current_time, price))
                # Remove data older than 1 hour (3600 seconds)
                price_history = [p for p in price_history if current_time - p[0] <= 3600]
                
                # 2. Check Price Move ($1 in 1 hour)
                start_price = price_history[0][1] # Oldest price in memory
                price_change = abs(price - start_price)
                is_volatile = price_change >= 1.0

                # 3. Get AI Analysis
                headlines = get_news()
                analysis = analyze_market(price, rsi, trend, headlines)
                chart = create_chart(data)

                # 4. Check Risk Level (Low/Medium)
                is_safe_risk = "Risk Level: Low" in analysis or "Risk Level: Medium" in analysis or "Risk Level: Med" in analysis
                
                # --- DECISION FILTER ---
                if is_safe_risk:
                    print("✅ Sending Alert: Good Risk Level.")
                    send_telegram(price, trend, analysis, chart, "Opportunity")
                
                elif is_volatile:
                    print(f"⚠️ Sending Alert: Big Move (${price_change:.2f})")
                    send_telegram(price, trend, analysis, chart, "Big Volatility")
                
                else:
                    print(f"zzz... Skipping. (Risk is High & Move is only ${price_change:.2f})")

            else:
                print("❌ No data received.")

        except Exception as e:
            print(f"⚠️ Crash prevention: {e}")
        
        # SLEEP FOR 10 MINUTES (600 seconds)
        print("⏳ Waiting 10 minutes...")
        time.sleep(660)
