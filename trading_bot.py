import os
import time
import calendar
import yfinance as yf
import requests
import feedparser
import pandas as pd
import numpy as np  # Required for the algorithmic channel math
import mplfinance as mpf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# --- 1. SETUP KEYS & SETTINGS ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# 🛑 SILENT MODE SETTING: 
# Set to True: Bot ONLY messages you if Conviction is >= 45% or an Emergency hits.
# Set to False: Bot messages you every 30 minutes, even if it says "STRICT WAIT".
MUTE_WAIT_SIGNALS = True 

# --- 2. GET DATA (15 DAYS / 1-HOUR MODE) ---
def get_market_data():
    ticker = "CL=F"
    try:
        # Pulling 15 days of 1-hour candles for accurate macro trends
        data = yf.download(ticker, period="15d", interval="1h", progress=False)
        if data.empty: return None, 0, 0, "No Data", 0, 0, 0, 0, 0
        
        # Clean multi-index columns if yfinance acts up
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.droplevel(1)
            
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close

        # TA Indicators
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
        
        # Calculate horizontal Support/Resistance for Gemini's brain
        plot_data = data.tail(150) # Look at the last ~6 days for immediate S/R
        recent_high = float(plot_data['High'].max())
        recent_low = float(plot_data['Low'].min())
        
        return data, price, rsi, trend, atr, ema50, ema21, recent_high, recent_low
    except Exception as e:
        print(f"Data Error: {e}")
        return None, 0, 0, "Error", 0, 0, 0, 0, 0

# --- 3. DRAW CHART (DARK MODE + VISIBLE CHANNELS) ---
def create_chart(data):
    if data is None: return None
    fname = "oil_chart.png"
    
    # We plot the last 150 hours to make the lines clearly visible
    plot_data = data.tail(150)
    
    # 1. Math for Horizontal Support & Resistance
    recent_high = plot_data['High'].max()
    recent_low = plot_data['Low'].min()
    horizontal_lines = [recent_high, recent_low]
    
    # 2. Math for Trendlines (Linear Regression Channel)
    x = np.arange(len(plot_data))
    y = plot_data['Close'].values
    
    # Calculate the slope (m) and intercept (b) of the main trend
    m, b = np.polyfit(x, y, 1)
    
    # Create the central regression line
    reg_line = m * x + b
    
    # Find the maximum distance from the center line to the highs and lows
    upper_offset = (plot_data['High'].values - reg_line).max()
    lower_offset = (reg_line - plot_data['Low'].values).max()
    
    # Create the top and bottom channel lines
    upper_channel = reg_line + upper_offset
    lower_channel = reg_line - lower_offset
    
    # Grab the start and end dates for the exact coordinates
    date_start = plot_data.index[0]
    date_end = plot_data.index[-1]
    
    # Create coordinate pairs for mplfinance: [(Date1, Price1), (Date2, Price2)]
    angled_lines = [
        [(date_start, lower_channel[0]), (date_end, lower_channel[-1])], # Support Floor
        [(date_start, upper_channel[0]), (date_end, upper_channel[-1])]  # Resistance Ceiling
    ]
    
    # Draw the chart using DARK MODE ('nightclouds') and thicker white lines
    mpf.plot(plot_data, type='candle', style='nightclouds', volume=False, mav=(21, 50), 
             hlines=dict(hlines=horizontal_lines, colors=['#00aaff', '#ffcc00'], linestyle='--'),
             alines=dict(alines=angled_lines, colors=['white', 'white'], linewidths=2.0),
             savefig=fname)
    return fname

# --- 4. GET NEWS (STRICT OIL-AFFECTED FILTER) ---
def get_news():
    try:
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

# --- 6. ANALYZE (1H CHART + S/R AWARENESS) ---
def analyze_market(price, rsi, trend, atr, ema50, ema21, recent_high, recent_low, headlines, alert_reason):
    model_name = get_valid_model()
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    current_time_str = time.strftime("%A, %b %d, %Y - %H:%M UTC", time.gmtime())
    
    stop_loss_buy = price - (1.5 * atr)
    take_profit_buy = price + (2.5 * atr)
    stop_loss_sell = price + (1.5 * atr) 
    take_profit_sell = price - (2.5 * atr)
    
    ema_status = "BULLISH (21 > 50)" if ema21 > ema50 else "BEARISH (21 < 50)"

    prompt = f"""
    Act as a Tactical Hedge Fund Algo (Day Trading Desk). 
    
    🚨 SYSTEM ALERT TRIGGER: {alert_reason} 🚨
    ⏰ CURRENT SYSTEM TIME: {current_time_str}
    
    GLOBAL NEWS FEED (WITH TIMESTAMPS):
    {news_text}
    
    TECHNICAL DATA (1-Hour Chart):
    - Price: ${price:.2f}
    - Major Resistance Ceiling: ${recent_high:.2f}
    - Major Support Floor: ${recent_low:.2f}
    - EMA Trend: {ema_status}
    - RSI: {rsi:.2f}
    - Volatility (ATR): {atr:.2f}
    
    TASK:
    You are a Tactical Day Trader. Your directive is to find actionable setups without overtrading.
    
    1. TIME-FILTER THE NEWS (CRITICAL): Compare the news timestamps to the CURRENT SYSTEM TIME. 
       - You MUST prioritize the absolute newest breaking headlines. 
       - IGNORE outdated historical narratives (e.g., old OPEC cuts, past wars) if they contradict today's live news feed.
    2. CALCULATE CONVICTION: Rate the setup from 0% to 100%.
       - For a TRADABLE score (45%+), the freshest breaking news should generally support the Technical Chart Trend, or the technicals must be overwhelmingly strong.
       - If the market is completely flat, RSI is entirely neutral, or the freshest news violently contradicts the chart, Conviction is LOW.
    3. ASSESS RISK LEVEL:
       - 85% to 100% = 🟢 LOW RISK (A+ Setup)
       - 60% to 84% = 🟡 MEDIUM RISK (Standard Setup)
       - 45% to 59% = 🔴 HIGH RISK (Aggressive/Early Entry)
    4. DECIDE ACTION: 
       - If Conviction is LESS THAN 45%, your Action MUST be "STRICT WAIT". 
       - If Conviction is 45% or higher, output "BUY" or "SELL".
    
    CALCULATED LIMITS:
    - BUY Setup: Stop=${stop_loss_buy:.2f}, Target=${take_profit_buy:.2f}
    - SELL Setup: Stop=${stop_loss_sell:.2f}, Target=${take_profit_sell:.2f}
    
    OUTPUT FORMAT (Strictly follow this):
    
    🎯 **CONVICTION SCORE: [0-100]%**
    ⚠️ **RISK LEVEL: [🟢 LOW RISK / 🟡 MEDIUM RISK / 🔴 HIGH RISK]**
    
    🌍 **MARKET DRIVER**
    [Identify the #1 freshest factor from the news. Explain in 1 sentence.]
    
    💎 **TRADE DECISION**
    Action: [BUY / SELL / STRICT WAIT]
    Entry: ${price:.2f}
    🛡️ Stop: [ATR Value]
    🎯 Target: [ATR Value]
    
    📊 **REASONING**
    - [Explain the conviction score and risk level. Focus on the TIMING of the news and how it matches the 1-Hour technicals, specifically mentioning if it is approaching the ${recent_high:.2f} Resistance or ${recent_low:.2f} Support.]
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
    
    header = "🚨 **EMERGENCY WTI ALERT** 🚨" if "SPIKE" in alert_reason or "BREAKING" in alert_reason else "🏎️ **WTI TACTICAL REPORT**"
    
    text = f"{header}\nTrigger: {alert_reason}\nPrice: ${price:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("🚀 Bot Started in Quant Mode (Dark Chart, Algo Trendlines, 1H Data)...")
    
    last_full_report_time = 0 
    last_price = 0
    seen_news_links = set()

    while True:
        try:
            print("Sentry checking market quietly...")
            data, price, rsi, trend, atr, ema50, ema21, recent_high, recent_low = get_market_data()
            headlines, raw_entries = get_news()
            
            if data is None:
                time.sleep(120)
                continue

            current_time = time.time()
            is_emergency = False
            alert_reason = "Regular 30-min Check"

            # 1. CHECK FOR PRICE SPIKE ($0.50 move)
            if last_price > 0 and abs(price - last_price) >= 0.50:
                is_emergency = True
                alert_reason = f"PRICE SPIKE! Moved ${abs(price - last_price):.2f} suddenly!"

            # 2. CHECK FOR BREAKING OIL NEWS (< 15 mins old)
            if not is_emergency: 
                current_utc = calendar.timegm(time.gmtime())
                for entry in raw_entries:
                    if 'published_parsed' in entry:
                        entry_time = calendar.timegm(entry.published_parsed)
                        age_in_seconds = current_utc - entry_time
                        
                        if age_in_seconds < 900 and entry.link not in seen_news_links:
                            is_emergency = True
                            alert_reason = f"BREAKING OIL NEWS: {entry.title.split(' - ')[0]}"
                            seen_news_links.add(entry.link)
                            break 

            # 3. CHECK IF 30 MINUTES HAVE PASSED
            time_since_last_report = current_time - last_full_report_time
            is_time_up = time_since_last_report >= 1800 

            # TRIGGER GEMINI
            if is_emergency or is_time_up:
                print(f"⚠️ Waking up Gemini! Reason: {alert_reason}")
                chart = create_chart(data)
                
                analysis = analyze_market(price, rsi, trend, atr, ema50, ema21, recent_high, recent_low, headlines, alert_reason)
                
                # --- SILENT MODE LOGIC ---
                is_wait_signal = "STRICT WAIT" in analysis.upper()
                
                if MUTE_WAIT_SIGNALS and is_wait_signal and not is_emergency:
                    print("🛑 AI says STRICT WAIT (< 45% Conviction). Muting Telegram to avoid spam.")
                else:
                    send_telegram(price, trend, analysis, chart, alert_reason)
                    print("✅ Report Sent to Telegram!")
                
                # Reset timers
                last_full_report_time = time.time()
                last_price = price
            else:
                print("Market is quiet. Sleeping for 2 minutes...")
                
        except Exception as e:
            print(f"⚠️ Crash prevention: {e}")
        
        time.sleep(120)
