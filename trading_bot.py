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

# --- 2. GET DATA (SPEED MODE: 21 & 50 EMA) ---
def get_market_data():
    ticker = "CL=F"
    try:
        # Download 5 Days of 30-min candles
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, "No Data", 0, 0, 0
        
        # Clean Data Format
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close

        # --- INDICATORS ---
        # 1. RSI
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        
        # 2. MACD
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        
        # 3. ATR (Volatility)
        data["ATR"] = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14).average_true_range()
        
        # 4. SPEED EMAs (For 1-Hour Trading)
        data["EMA50"] = EMAIndicator(close=data["Close"], window=50).ema_indicator() # The Trend
        data["EMA21"] = EMAIndicator(close=data["Close"], window=21).ema_indicator() # The Trigger
        
        # Get Latest Values
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
    # Show the Fast (21) and Medium (50) EMA
    mpf.plot(data.tail(50), type='candle', style='charles', volume=False, mav=(21, 50), savefig=fname)
    return fname

# --- 4. GET NEWS (MODIFIED: GLOBAL DRAGNET) ---
def get_news():
    try:
        # THE "CATCH-ALL" QUERY for Economy, War, Disasters, and Supply
        query = (
            "Crude Oil OR Energy Market OR OPEC OR Iran OR Russia OR Ukraine OR "
            "Federal Reserve OR Inflation OR China Economy OR "
            "Hurricane OR Earthquake OR Refinery Fire OR Pipeline Burst OR Sanctions"
        )
        
        # Format the URL safely with the massive query
        base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
        final_url = base_url.format(requests.utils.quote(query))
        
        feed = feedparser.parse(final_url)
        if not feed.entries: return []
        
        # Grab top 10 headlines to ensure we don't miss hidden events
        headlines = []
        for entry in feed.entries[:10]:
            pub_time = entry.get('published', '')
            # Clean title to remove extra source tags at the end
            title = entry.title.split(" - ")[0]
            headlines.append(f"{title} ({pub_time})")
            
        return headlines
    except:
        return []

# --- 5. FIND MODEL (THE WORKING FIX) ---
def get_valid_model():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}"
    try:
        # Dynamically ask Google which models are available
        resp = requests.get(url)
        data = resp.json()
        if 'models' in data:
            for model in data['models']:
                if 'generateContent' in model.get('supportedGenerationMethods', []):
                    # Prefer 1.5-flash if available, but take what works
                    if "flash" in model['name']:
                        return model['name']
            # Fallback to the first available one if Flash isn't found
            return data['models'][0]['name']
    except:
        pass
    # Absolute backup
    return "models/gemini-1.5-flash"

# --- 6. ANALYZE (MODIFIED: MACRO-ECONOMIST LOGIC) ---
def analyze_market(price, rsi, trend, atr, ema50, ema21, headlines):
    model_name = get_valid_model()
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    # --- MATH: Day Trading Stops (1.5x ATR) ---
    stop_loss_buy = price - (1.5 * atr)
    take_profit_buy = price + (2.5 * atr)
    stop_loss_sell = price + (1.5 * atr) 
    take_profit_sell = price - (2.5 * atr)
    
    ema_status = "BULLISH (21 > 50)" if ema21 > ema50 else "BEARISH (21 < 50)"

    # MACRO-ECONOMIST PROMPT
    prompt = f"""
    Act as a Wall Street Strategist & Algo Trader.
    
    GLOBAL NEWS FEED (ECONOMY + WAR + SUPPLY):
    {news_text}
    
    TECHNICAL DATA (30-min Chart):
    - Price: ${price:.2f}
    - EMA Trend: {ema_status}
    - RSI: {rsi:.2f}
    - Volatility (ATR): {atr:.2f}
    
    TASK:
    1. FILTER NEWS: Ignore noise. Focus on the ONE event moving the market TODAY.
       - Is it War/Geopolitics? (Bullish)
       - Is it a Strong Dollar/Fed Rates/Inflation? (Bearish)
       - Is it China slowing down? (Bearish)
       - Is it a Hurricane/Refinery outage? (Bullish)
    2. CORRELATE: Cross-reference the dominant News with the Technical Chart. 
       - If News is Bearish but Chart says BUY -> ISSUE "WAIT".
       - If News & Chart agree -> ISSUE STRONG SIGNAL.
    3. Use the calculated limits below for Risk Management.
    
    CALCULATED LIMITS (Day Trade):
    - BUY Setup: Stop=${stop_loss_buy:.2f}, Target=${take_profit_buy:.2f}
    - SELL Setup: Stop=${stop_loss_sell:.2f}, Target=${take_profit_sell:.2f}
    
    OUTPUT FORMAT (Strictly follow this):
    
    🌍 **MARKET DRIVER**
    [Identify the #1 factor from the news: War, Economy, or Supply? Explain in 1 sentence.]
    
    💎 **TRADE DECISION**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛡️ Stop: [ATR Value]
    🎯 Target: [ATR Value]
    
    📊 **REASONING**
    - [Macro Analysis: How does the news affect Global Demand/Supply?]
    - [Technical Confirmation]
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
    
    text = f"🏎️ **WTI SPEED REPORT (Macro Mode)**\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP (RUNS FOREVER) ---
if __name__ == "__main__":
    print("🚀 Bot Started in Speed Mode (Macro Dragnet)...")
    while True:
        try:
            print("Analyzing market...")
            # Unpack the 7 values (ATR + EMA50 + EMA21)
            data, price, rsi, trend, atr, ema50, ema21 = get_market_data()
            
            if data is not None:
                chart = create_chart(data)
                headlines = get_news()
                # Pass the 7 values to analysis
                analysis = analyze_market(price, rsi, trend, atr, ema50, ema21, headlines)
                send_telegram(price, trend, analysis, chart)
                print("✅ Report Sent!")
            else:
                print("❌ No data received.")
        except Exception as e:
            print(f"⚠️ Crash prevention: {e}")
        
        # 30 Minute Sleep
        print("💤 Sleeping for 30 minutes...")
        time.sleep(1800)
