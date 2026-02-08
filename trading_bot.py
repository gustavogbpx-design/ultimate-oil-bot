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

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- ASSETS ---
ASSETS = {
    "WTI Oil": "CL=F",
    "Gold": "GC=F"
}

# --- DATA FETCHING ---
def get_market_data(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, 0, 0, "No Signal"
        
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close
        
        # Indicators
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        data["EMA200"] = EMAIndicator(close=data["Close"], window=200).ema_indicator()
        data["ATR"] = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14).average_true_range()
        
        price = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        atr = data["ATR"].iloc[-1]
        ema200 = data["EMA200"].iloc[-1]
        trend = "BULLISH 🟢" if data["MACD"].iloc[-1] > data["Signal"].iloc[-1] else "BEARISH 🔴"
        
        return data, price, rsi, trend, atr, ema200
        
    except Exception as e:
        print(f"Data Error ({ticker}): {e}")
        return None, 0, 0, "Error", 0, 0

# --- CHART ---
def create_chart(data, asset_name):
    if data is None: return None
    fname = f"{asset_name.replace(' ', '_')}_chart.png"
    mpf.plot(data.tail(50), type='candle', style='charles', volume=False, 
             mav=(50, 200), title=f"{asset_name} (Training Mode)", savefig=fname)
    return fname

# --- NEWS ---
def get_news(asset_name):
    search_term = "Crude+Oil" if "Oil" in asset_name else "Gold+Price"
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={search_term}&hl=en-US&gl=US&ceid=US:en")
        if not feed.entries: return []
        return [entry.title for entry in feed.entries[:3]]
    except:
        return []

# --- AI ANALYSIS (DEBUG VERSION) ---
def analyze_market(asset, price, rsi, trend, atr, ema200, headlines):
    
    stop_loss_buy = price - (2.0 * atr)
    stop_loss_sell = price + (2.0 * atr)
    ema_status = "ABOVE 200 EMA" if price > ema200 else "BELOW 200 EMA"
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    prompt = f"""
    Act as a Hedge Fund Trader in TRAINING MODE.
    
    ASSET: {asset}
    - Price: ${price:.2f}
    - RSI: {rsi:.2f}
    - Trend: {trend}
    - EMA: {ema_status}
    - ATR: {atr:.2f}
    
    NEWS:
    {news_text}
    
    TASK:
    Analyze the current setup. Even if it is bad, tell me the best possible move.
    
    OUTPUT FORMAT:
    💎 **{asset.upper()} UPDATE**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛡️ Stop (ATR): ${stop_loss_buy:.2f} (Buy) / ${stop_loss_sell:.2f} (Sell)
    
    📊 **ANALYSIS**
    Risk Level: [Low/Med/High]
    Reasoning: [Short explanation]
    """
    
    try:
        # Check if Key is missing immediately
        if not GEMINI_KEY:
            return "⚠️ CRITICAL ERROR: API Key is missing! Check Railway Variables."

        model_name = "models/gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_KEY}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        
        # --- NEW DEBUGGING LOGIC ---
        if resp.status_code == 200:
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            # This will show the EXACT error from Google in your Telegram
            return f"⚠️ GOOGLE ERROR {resp.status_code}:\n{resp.text}"
            
    except Exception as e:
        return f"⚠️ PYTHON ERROR: {e}"

# --- SEND TELEGRAM ---
def send_telegram(asset, price, analysis, chart_file):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    
    text = f"🏋️ **DEBUG REPORT ({asset})**\nPrice: ${price:.2f}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("🚀 Debug Mode Started...")
    
    while True:
        try:
            for asset_name, ticker in ASSETS.items():
                print(f"🔍 Checking {asset_name}...")
                data, price, rsi, trend, atr, ema200 = get_market_data(ticker)
                
                if data is not None:
                    headlines = get_news(asset_name)
                    chart = create_chart(data, asset_name)
                    analysis = analyze_market(asset_name, price, rsi, trend, atr, ema200, headlines)

                    print(f"✅ Sending report for {asset_name}...")
                    send_telegram(asset_name, price, analysis, chart)
                
                time.sleep(5) 

        except Exception as e:
            print(f"⚠️ Error: {e}")
        
        print("⏳ Waiting 10 minutes...")
        time.sleep(600)
