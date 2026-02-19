import os
import time
import calendar
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

        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        data["ATR"] = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14).average_true_range()
        
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

# --- 4. GET NEWS (STRICT OIL-AFFECTED FILTER) ---
def get_news():
    try:
        # THE DOUBLE FILTER:
        # Part 1: Direct Oil news (Crude Oil, WTI, OPEC)
        # Part 2: Macro Events (War, Iran, Russia, Earthquakes) MUST also contain "Oil" or "Energy"
        query = "(\"Crude Oil\" OR \"WTI\" OR OPEC) OR ((\"War\" OR \"Attack\" OR \"Iran\" OR \"Russia\" OR \"Ukraine\" OR \"Explosion\" OR \"Refinery\" OR \"Sanctions\" OR \"Hurricane\") AND (\"Oil\" OR \"Energy\"))"
        
        base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
        final_url = base_url.format(requests.utils.quote(query))
        
        feed = feedparser.parse(final_url)
        if not feed.entries: return [], []
        
        headlines = []
        raw_entries = []
        for entry in feed.entries[:10]:
            pub_time = entry.get('published', '')
            title = entry.title.split(" - ")[0]
            headlines.append(f"{title} ({pub_time})")
            raw_entries.append(entry)
            
        return headlines, raw_entries
    except:
        return [], []

# --- 5. FIND MODEL ---
def get_valid_model():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}"
    try:
        resp = requests.get(url)
        data = resp.json()
        if 'models' in data:
            for model in data['models']:
                if 'generateContent' in model.get('supportedGenerationMethods', []):
                    if "flash" in model['name']: return model['name']
            return data['models'][0]['name']
    except: pass
    return "models/gemini-1.5-flash"

# --- 6. ANALYZE (GEMINI BRAIN) ---
def analyze_market(price, rsi, trend, atr, ema50, ema21, headlines, alert_reason):
    model_name = get_valid_model()
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    stop_loss_buy = price - (1.5 * atr)
    take_profit_buy = price + (2.5 * atr)
    stop_loss_sell = price + (1.5 * atr) 
    take_profit_sell = price - (2.5 * atr)
    
    ema_status = "BULLISH (21 > 50)" if ema21 > ema50 else "BEARISH (21 < 50)"

    prompt = f"""
    Act as a Wall Street Strategist. 
    
    🚨 SYSTEM ALERT TRIGGER: {alert_reason} 🚨
    (If this says "Regular 30-min Check", do standard analysis. If it says "PRICE SPIKE" or "BREAKING NEWS", focus heavily on explaining the sudden emergency).
    
    GLOBAL NEWS FEED:
    {news_text}
    
    TECHNICAL DATA:
    - Price: ${price:.2f}
    - EMA Trend: {ema_status}
    - RSI: {rsi:.2f}
    - Volatility (ATR): {atr:.2f}
    
    TASK:
    1. FILTER NEWS: Identify the #1 market driver right now.
    2. CORRELATE: Cross-reference the dominant News with the Technical Chart. 
    3. Use the calculated limits below for Risk Management.
    
    CALCULATED LIMITS:
    - BUY Setup: Stop=${stop_loss_buy:.2f}, Target=${take_profit_buy:.2f}
    - SELL Setup: Stop=${stop_loss_sell:.2f}, Target=${take_profit_sell:.2f}
    
    OUTPUT FORMAT (Strictly follow this):
    
    🌍 **MARKET DRIVER**
    [Identify the #1 factor from the news. Explain in 1 sentence.]
    
    💎 **TRADE DECISION**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛡️ Stop: [ATR Value]
    🎯 Target: [ATR Value]
    
    📊 **REASONING**
    - [Macro Analysis]
    - [Technical Confirmation]
    """

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_KEY}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        if resp.status_code == 200:
            ai_reply = resp.json()['candidates'][0]['content']['parts'][0]['text']
            return f"🧠 **AI SIGNAL ({model_name.split('/')[-1]}):**\n{ai_reply}"
        else: return f"⚠️ AI Error: {resp.status_code} - {resp.text}"
    except Exception as e: return f"⚠️ AI Connection Failed: {e}"

# --- 7. SEND TELEGRAM ---
def send_telegram(price, trend, analysis, chart_file, alert_reason):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    
    # Add alarm emojis if it's an emergency
    header = "🚨 **EMERGENCY WTI ALERT** 🚨" if "SPIKE" in alert_reason or "BREAKING" in alert_reason else "🏎️ **WTI SPEED REPORT**"
    
    text = f"{header}\nTrigger: {alert_reason}\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP (SENTRY MODE) ---
if __name__ == "__main__":
    print("🚀 Bot Started in Strict Sentry Mode (2-min watch, 30-min Gemini)...")
    
    last_full_report_time = 0 
    last_price = 0
    seen_news_links = set() # Bot's memory of news it already told you about

    while True:
        try:
            print("Sentry checking market quietly...")
            data, price, rsi, trend, atr, ema50, ema21 = get_market_data()
            headlines, raw_entries = get_news()
            
            if data is None:
                time.sleep(120)
                continue

            current_time = time.time()
            is_emergency = False
            alert_reason = "Regular 30-min Check"

            # --- 1. CHECK FOR PRICE SPIKE (More than $0.50 move) ---
            if last_price > 0 and abs(price - last_price) >= 0.50:
                is_emergency = True
                alert_reason = f"PRICE SPIKE! Moved ${abs(price - last_price):.2f} suddenly!"

            # --- 2. CHECK FOR BREAKING OIL NEWS (< 15 mins old) ---
            if not is_emergency: 
                current_utc = calendar.timegm(time.gmtime())
                for entry in raw_entries:
                    if 'published_parsed' in entry:
                        entry_time = calendar.timegm(entry.published_parsed)
                        age_in_seconds = current_utc - entry_time
                        
                        # Trigger if news is < 15 mins (900 seconds) AND hasn't been seen yet
                        if age_in_seconds < 900 and entry.link not in seen_news_links:
                            is_emergency = True
                            alert_reason = f"BREAKING OIL NEWS: {entry.title.split(' - ')[0]}"
                            seen_news_links.add(entry.link)
                            break 

            # --- 3. CHECK IF 30 MINUTES HAVE PASSED ---
            time_since_last_report = current_time - last_full_report_time
            is_time_up = time_since_last_report >= 1800 

            # --- TRIGGER GEMINI ---
            if is_emergency or is_time_up:
                print(f"⚠️ Waking up Gemini! Reason: {alert_reason}")
                chart = create_chart(data)
                
                analysis = analyze_market(price, rsi, trend, atr, ema50, ema21, headlines, alert_reason)
                send_telegram(price, trend, analysis, chart, alert_reason)
                
                print("✅ Report Sent!")
                
                last_full_report_time = time.time()
                last_price = price
            else:
                print("Market is quiet. Sleeping for 2 minutes...")
                
        except Exception as e:
            print(f"⚠️ Crash prevention: {e}")
        
        time.sleep(120)
