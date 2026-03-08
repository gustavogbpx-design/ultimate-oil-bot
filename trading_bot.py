import os
import time
import calendar
import yfinance as yf
import requests
import feedparser
import pandas as pd
import numpy as np
import mplfinance as mpf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# --- 1. SETUP KEYS & SETTINGS ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MUTE_WAIT_SIGNALS = True 

# --- UPGRADED MATH ENGINE: UNIVERSAL LINEAR REGRESSION CHANNEL ---
def calculate_universal_channel(slice_data, cutoff_pct=0.92):
    """Dynamically calculates an upward or downward channel for ANY data slice."""
    if len(slice_data) < 20:
        return False, [], []
        
    x = np.arange(len(slice_data))
    y = slice_data['Close'].values
    
    # 1. Find the universal Line of Best Fit (Works for Up, Down, or Flat trends)
    # m = slope (direction), b = intercept (starting height)
    m, b = np.polyfit(x, y, 1)
    
    # Create the center mathematical line
    reg_line = m * x + b
    
    # 2. Calculate offsets for all Highs and Lows from the center line
    high_offsets = slice_data['High'].values - reg_line
    low_offsets = reg_line - slice_data['Low'].values
    
    # 3. Apply the Percentile Filter to BOTH sides (Ignores flash crashes and breakout spikes)
    sorted_highs = np.sort(high_offsets)
    sorted_lows = np.sort(low_offsets)
    
    if len(sorted_highs) > 0 and len(sorted_lows) > 0:
        # Prevent index out of bounds
        cutoff_idx = int(len(sorted_highs) * cutoff_pct)
        cutoff_idx = min(cutoff_idx, len(sorted_highs) - 1)
        
        upper_offset = sorted_highs[cutoff_idx]
        lower_offset = sorted_lows[cutoff_idx]
    else:
        upper_offset, lower_offset = 1.0, 1.0
        
    # 4. Create the final parallel channel boundaries
    upper_channel = reg_line + upper_offset
    lower_channel = reg_line - lower_offset
    
    date_start = slice_data.index[0]
    date_end = slice_data.index[-1]
    
    # Return coordinate pairs for mplfinance
    support_line = [(date_start, lower_channel[0]), (date_end, lower_channel[-1])]
    resistance_line = [(date_start, upper_channel[0]), (date_end, upper_channel[-1])]
    
    return True, support_line, resistance_line

# --- 2. GET DATA (25 DAYS / 1-HOUR MODE) ---
def get_market_data():
    ticker = "CL=F"
    try:
        data = yf.download(ticker, period="25d", interval="1h", progress=False)
        if data.empty: return None, 0, 0, "No Data", 0, 0, 0, 0, None
        
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.droplevel(1)
            
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
        recent_low = float(data.tail(250)['Low'].min())
        
        return data, price, rsi, trend, atr, ema50, ema21, recent_low, data.tail(250)
    except Exception as e:
        print(f"Data Error: {e}")
        return None, 0, 0, "Error", 0, 0, 0, 0, None

# --- 3. DRAW CHART (UNIVERSAL DUAL-CHANNEL MODE) ---
def create_chart(plot_data):
    if plot_data is None: return None
    fname = "oil_chart.png"
    
    recent_low = plot_data['Low'].min()
    horizontal_lines = [recent_low] 
    
    # --- CALCULATE UNIVERSAL CHANNELS ---
    # 1. Macro Channel: Full 25-day trend, filtering out top/bottom 8% of wicks
    macro_exists, macro_sup, macro_res = calculate_universal_channel(plot_data, cutoff_pct=0.92)
    
    # 2. Micro Channel: Last 70 hours, filtering top/bottom 20% to hug the candles tightly
    micro_exists, micro_sup, micro_res = calculate_universal_channel(plot_data.tail(70), cutoff_pct=0.80)

    mc = mpf.make_marketcolors(up='#00E676', down='#D500F9', edge='inherit', wick='inherit', volume='in')
    iq_style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)
    
    kwargs = dict(type='candle', style=iq_style, volume=False, mav=(21, 50), 
                  hlines=dict(hlines=horizontal_lines, colors=['#ffcc00'], linestyle='--'),
                  savefig=fname)
                  
    lines_to_draw = []
    line_colors = []
    line_styles = []
    
    if macro_exists:
        lines_to_draw.extend([macro_sup, macro_res])
        line_colors.extend(['white', 'gray'])
        line_styles.extend(['-', '-']) 
        
    if micro_exists:
        lines_to_draw.extend([micro_sup, micro_res])
        line_colors.extend(['#00aaff', '#00aaff']) 
        line_styles.extend(['--', '--']) 

    if lines_to_draw:
        kwargs['alines'] = dict(alines=lines_to_draw, colors=line_colors, linestyle=line_styles, linewidths=2.0)

    mpf.plot(plot_data, **kwargs)
    return fname

# --- 4. GET NEWS ---
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

# --- 6. ANALYZE (UNIVERSAL CHANNEL AWARENESS) ---
def analyze_market(price, rsi, trend, atr, ema50, ema21, recent_low, headlines, alert_reason):
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
    - Absolute Hard Floor Support: ${recent_low:.2f}
    - Market Structure: UNIVERSAL DUAL-CHANNEL TRACKING ACTIVE (Macro & Micro trends being mapped via Linear Regression)
    - EMA Trend: {ema_status}
    - RSI: {rsi:.2f}
    - Volatility (ATR): {atr:.2f}
    
    TASK:
    You are a Tactical Day Trader. Your directive is to find actionable setups without overtrading.
    
    1. TIME-FILTER THE NEWS (CRITICAL): Compare the news timestamps to the CURRENT SYSTEM TIME. 
    2. CALCULATE CONVICTION: Rate the setup from 0% to 100%. 
    3. ASSESS RISK LEVEL: [🟢 LOW / 🟡 MEDIUM / 🔴 HIGH]
    4. DECIDE ACTION: [BUY / SELL / STRICT WAIT]
    
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
    - [Explain your reasoning based on the latest data.]
    """

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_KEY}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        if resp.status_code == 200:
            return f"🧠 **AI SIGNAL ({model_name.split('/')[-1]}):**\n{resp.json()['candidates'][0]['content']['parts'][0]['text']}"
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
    print("🚀 Bot Started in Quant Mode (Universal Linear Regression Engine)...")
    
    last_full_report_time = 0 
    last_price = 0
    seen_news_links = set()

    while True:
        try:
            print("Sentry checking market quietly...")
            data, price, rsi, trend, atr, ema50, ema21, recent_low, plot_data = get_market_data()
            headlines, raw_entries = get_news()
            
            if data is None:
                time.sleep(120)
                continue

            current_time = time.time()
            is_emergency = False
            alert_reason = "Regular 30-min Check"

            if last_price > 0 and abs(price - last_price) >= 0.50:
                is_emergency = True
                alert_reason = f"PRICE SPIKE! Moved ${abs(price - last_price):.2f} suddenly!"

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

            time_since_last_report = current_time - last_full_report_time
            is_time_up = time_since_last_report >= 1800 

            if is_emergency or is_time_up:
                print(f"⚠️ Waking up Gemini! Reason: {alert_reason}")
                chart = create_chart(plot_data)
                
                analysis = analyze_market(price, rsi, trend, atr, ema50, ema21, recent_low, headlines, alert_reason)
                
                is_wait_signal = "STRICT WAIT" in analysis.upper()
                
                if MUTE_WAIT_SIGNALS and is_wait_signal and not is_emergency:
                    print("🛑 AI says STRICT WAIT (< 45% Conviction). Muting Telegram to avoid spam.")
                else:
                    send_telegram(price, trend, analysis, chart, alert_reason)
                    print("✅ Report Sent to Telegram!")
                
                last_full_report_time = time.time()
                last_price = price
            else:
                print("Market is quiet. Sleeping for 2 minutes...")
                
        except Exception as e:
            print(f"⚠️ Crash prevention: {e}")
        
        time.sleep(120)
