import os
import time
import requests
import feedparser
import pandas as pd
import mplfinance as mpf
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange
import json

# --- 1. SETUP KEYS (FAILOVER SYSTEM) ---
# FIXED: Matches your Railway Environment Variables exactly
GEMINI_KEY_1 = os.environ.get("GEMINI_KEY_1") 
GEMINI_KEY_2 = os.environ.get("GEMINI_KEY_2") 

# We put them in a list to loop through them
API_KEYS = [key for key in [GEMINI_KEY_1, GEMINI_KEY_2] if key]

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- 2. GET DATA ---
def get_market_data():
    ticker = "CL=F"
    try:
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, "No Data", 0, 0
        
        # Clean Data Format
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close

        # Indicators
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        data["ATR"] = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14).average_true_range()
        data["EMA200"] = EMAIndicator(close=data["Close"], window=200).ema_indicator()
        
        # Latest Values
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
    mpf.plot(data.tail(50), type='candle', style='charles', mav=(50, 200), savefig=fname)
    return fname

# --- 4. GET NEWS ---
def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Iran+Conflict&hl=en-US&gl=US&ceid=US:en")
        if not feed.entries: return []
        return [entry.title for entry in feed.entries[:5]]
    except:
        return []

# --- 5. FIND MODEL HELPER ---
def get_model_url(api_key):
    # This function constructs the URL for a specific key
    return f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

# --- 6. ANALYZE (WITH KEY ROTATION) ---
def analyze_market(price, rsi, trend, atr, ema200, headlines):
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    # Calculate Stops
    stop_loss_buy = price - (2.0 * atr)
    take_profit_buy = price + (3.0 * atr)
    stop_loss_sell = price + (2.0 * atr) 
    take_profit_sell = price - (3.0 * atr)
    ema_status = "Price is ABOVE 200 EMA (Uptrend)" if price > ema200 else "Price is BELOW 200 EMA (Downtrend)"

    prompt = f"""
    Act as a Senior Wall Street Trader.
    MARKET DATA: Price: ${price:.2f}, RSI: {rsi:.2f}, Trend: {trend}, Context: {ema_status}, ATR: {atr:.2f}
    SMART STOPS: BUY(Stop=${stop_loss_buy:.2f}, Target=${take_profit_buy:.2f}) | SELL(Stop=${stop_loss_sell:.2f}, Target=${take_profit_sell:.2f})
    NEWS: {news_text}
    
    TASK: Determine best trade setup using calculated stops.
    OUTPUT FORMAT:
    💎 **TRADE SETUP**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛡️ Smart Stop: [Use ATR Value]
    🎯 Smart Target: [Use ATR Value]
    
    📊 **DEEP ANALYSIS**
    Risk Level: [Low/Med/High]
    Reasoning: [Technicals + News]
    """

    # --- KEY ROTATION LOGIC ---
    for i, current_key in enumerate(API_KEYS):
        try:
            print(f"🤖 Attempting AI analysis with Key #{i+1}...")
            url = get_model_url(current_key)
            
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            headers = {'Content-Type': 'application/json'}
            
            resp = requests.post(url, json=payload, headers=headers)
            
            # If Successful (200 OK)
            if resp.status_code == 200:
                ai_reply = resp.json()['candidates'][0]['content']['parts'][0]['text']
                return f"🧠 **AI SIGNAL (Key #{i+1}):**\n{ai_reply}"
            
            # If Quota Error (429) or Server Error (5xx)
            elif resp.status_code in [429, 500, 503]:
                print(f"⚠️ Key #{i+1} Failed (Status {resp.status_code}). Switching to backup...")
                continue # Loop tries the next key
            
            # If other error (like 400 Bad Request), don't retry, just fail
            else:
                return f"⚠️ API Error (Key #{i+1}): {resp.text}"

        except Exception as e:
            print(f"⚠️ Connection Error on Key #{i+1}: {e}")
            continue # Try next key

    return "❌ All API Keys failed. Quota exceeded on both accounts."

# --- 7. SEND TELEGRAM ---
def send_telegram(price, trend, analysis, chart_file):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    
    # 1. Send Chart
    if chart_file and os.path.exists(chart_file):
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    
    # 2. Send Text
    # We split long messages to avoid Telegram limits
    msg = f"🛢 **WTI MASTER REPORT**\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    
    if len(msg) > 4000:
        # Split if too long
        requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg[:4000]})
        requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg[4000:]})
    else:
        requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg})

# --- MAIN LOOP ---
if __name__ == "__main__":
    if not API_KEYS:
        print("❌ CRITICAL ERROR: No Gemini API Keys found. Check your environment variables.")
        # We don't exit here anymore to prevent the container from crashing loops
        # It will just print errors until you fix the variables
        
    print(f"🚀 Bot Started using {len(API_KEYS)} API Keys for Redundancy.")
    
    while True:
        try:
            print(f"\n⏰ Time: {time.strftime('%H:%M:%S')} - Analyzing market...")
            data, price, rsi, trend, atr, ema200 = get_market_data()
            
            if data is not None:
                chart = create_chart(data)
                headlines = get_news()
                analysis = analyze_market(price, rsi, trend, atr, ema200, headlines)
                send_telegram(price, trend, analysis, chart)
                print("✅ Report Sent!")
            else:
                print("❌ No data received.")
        except Exception as e:
            print(f"⚠️ Crash prevention: {e}")
        
        # Sleep 10 mins
        print("💤 Sleeping for 10 minutes...")
        time.sleep(600)
