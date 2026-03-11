import os
import time
import calendar
import base64 
import threading 
import re # NEW: To extract data for the UI
import yfinance as yf
import requests
import feedparser
import pandas as pd
import numpy as np
import mplfinance as mpf
from flask import Flask, render_template_string, send_file 
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# --- 1. SETUP KEYS & SETTINGS ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MUTE_WAIT_SIGNALS = True 

# --- GLOBAL STATE FOR THE DASHBOARD ---
DASHBOARD_DATA = {
    "price": 0.00,
    "trend": "BOOTING...",
    "analysis": "Awaiting initial quant calculation...",
    "news": [],
    "last_update": "N/A",
    "status": "Scanning...",
    "rsi": 0.00,
    "atr": 0.00,
    "ema_status": "WAITING",
    "conviction": "0%",
    "risk": "WAITING",
    "trade_action": "STRICT WAIT",
    "trade_entry": 0.00,
    "trade_stop": 0.00,
    "trade_target": 0.00
}

# --- WEB SERVER SETUP ---
app = Flask(__name__)

# --- UPGRADED MATH ENGINE: UNIVERSAL LINEAR REGRESSION CHANNEL ---
def calculate_universal_channel(slice_data, cutoff_pct=0.92):
    if len(slice_data) < 20: return False, [], []
    x = np.arange(len(slice_data))
    y = slice_data['Close'].values
    m, b = np.polyfit(x, y, 1)
    reg_line = m * x + b
    high_offsets = slice_data['High'].values - reg_line
    low_offsets = reg_line - slice_data['Low'].values
    sorted_highs = np.sort(high_offsets)
    sorted_lows = np.sort(low_offsets)
    
    if len(sorted_highs) > 0 and len(sorted_lows) > 0:
        cutoff_idx = int(len(sorted_highs) * cutoff_pct)
        cutoff_idx = min(cutoff_idx, len(sorted_highs) - 1)
        upper_offset = sorted_highs[cutoff_idx]
        lower_offset = sorted_lows[cutoff_idx]
    else:
        upper_offset, lower_offset = 1.0, 1.0
        
    upper_channel = reg_line + upper_offset
    lower_channel = reg_line - lower_offset
    date_start = slice_data.index[0]
    date_end = slice_data.index[-1]
    support_line = [(date_start, lower_channel[0]), (date_end, lower_channel[-1])]
    resistance_line = [(date_start, upper_channel[0]), (date_end, upper_channel[-1])]
    return True, support_line, resistance_line

# --- 2. GET DATA (25 DAYS / 1-HOUR MODE) ---
def get_market_data():
    ticker = "CL=F" # WTI Crude Oil
    try:
        data = yf.download(ticker, period="25d", interval="1h", progress=False)
        if data.empty: return None, 0, 0, "No Data", 0, 0, 0, 0, None
        
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
    
    macro_exists, macro_sup, macro_res = calculate_universal_channel(plot_data, cutoff_pct=0.92)
    micro_exists, micro_sup, micro_res = calculate_universal_channel(plot_data.tail(70), cutoff_pct=0.80)

    mc = mpf.make_marketcolors(up='#00E676', down='#D500F9', edge='inherit', wick='inherit', volume='in')
    iq_style = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, facecolor='#0f172a', edgecolor='#1e293b', figcolor='#0f172a')
    
    kwargs = dict(type='candle', style=iq_style, volume=False, mav=(21, 50), 
                  hlines=dict(hlines=horizontal_lines, colors=['#ffcc00'], linestyle='--'),
                  savefig=fname)
                  
    lines_to_draw, line_colors, line_styles = [], [], []
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

# --- 4. GET NEWS (STRICT 48-HOUR FRESHNESS FILTER) ---
def get_news():
    try:
        query = '("Crude Oil" OR "WTI" OR "OPEC" OR "Middle East") when:1d'
        base_url = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
        final_url = base_url.format(requests.utils.quote(query))
        feed = feedparser.parse(final_url)
        if not feed.entries: return [], []
        
        headlines, raw_entries = [], []
        current_time = calendar.timegm(time.gmtime())
        
        for entry in feed.entries:
            if 'published_parsed' in entry:
                entry_time = calendar.timegm(entry.published_parsed)
                age_in_hours = (current_time - entry_time) / 3600
                
                if age_in_hours <= 48:
                    pub_time = entry.get('published', '')
                    title = entry.title.split(" - ")[0]
                    headlines.append({"title": title, "time": pub_time, "link": entry.link})
                    raw_entries.append(entry)
            if len(headlines) >= 7: break
                
        return headlines, raw_entries
    except Exception as e:
        print(f"News fetch error: {e}")
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

# --- 6. ANALYZE (SNIPER MODE) ---
def analyze_market(price, rsi, trend, atr, ema50, ema21, recent_low, headlines, alert_reason, chart_file):
    model_name = get_valid_model()
    news_text = "\n".join([f"- {h['title']}" for h in headlines])
    current_time_str = time.strftime("%A, %b %d, %Y - %H:%M UTC", time.gmtime())
    
    stop_loss_buy = price - (1.5 * atr)
    take_profit_buy = price + (2.5 * atr)
    stop_loss_sell = price + (1.5 * atr) 
    take_profit_sell = price - (2.5 * atr)
    ema_status = "BULLISH (21 > 50)" if ema21 > ema50 else "BEARISH (21 < 50)"

    prompt = f"""
    Act as a Tactical Hedge Fund Algo (Day Trading Desk). 
    
    🚨 SYSTEM ALERT: {alert_reason}
    ⏰ TIME: {current_time_str}
    
    NEWS FEED:
    {news_text}
    
    TECHNICALS:
    - Price: ${price:.2f}
    - Floor Support: ${recent_low:.2f}
    - EMA Trend: {ema_status}
    - RSI: {rsi:.2f}
    - ATR: {atr:.2f}
    
    IMAGE DIRECTIVE: Look at the attached chart. White/Grey = Macro Channel. Blue Dashed = Micro Channel. 
    
    TASK: Give a highly concise, sniper-style quant report. NO fluff. NO long paragraphs. 
    
    CALCULATED LIMITS:
    - BUY: Stop=${stop_loss_buy:.2f}, Target=${take_profit_buy:.2f}
    - SELL: Stop=${stop_loss_sell:.2f}, Target=${take_profit_sell:.2f}
    
    OUTPUT FORMAT (STRICTLY FOLLOW THIS EXACT TEMPLATE):
    
    🎯 **CONVICTION SCORE:** [0-100]%
    ⚠️ **RISK LEVEL:** [🟢 LOW / 🟡 MEDIUM / 🔴 HIGH]
    
    🌍 **MARKET DRIVER**
    [Maximum 1 short sentence identifying the primary catalyst from the news.]
    
    👁️ **CHART VISION ANALYSIS**
    [Maximum 2 short sentences describing exactly where the price is interacting with the White/Grey/Blue lines.]
    
    💎 **TRADE DECISION**
    Action: [BUY / SELL / STRICT WAIT]
    Entry: ${price:.2f}
    🛡️ Stop: [Insert Calculated Limit]
    🎯 Target: [Insert Calculated Limit]
    
    📊 **REASONING**
    - *Fundamental:* [1 short sentence on the news impact.]
    - *Structure:* [1 short sentence on channel support/resistance.]
    - *Momentum:* [1 short sentence on RSI/EMA.]
    - *Risk:* [1 short sentence on why the stop loss is placed there.]
    """

    parts = [{"text": prompt}]
    if chart_file and os.path.exists(chart_file):
        with open(chart_file, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"inline_data": {"mime_type": "image/png", "data": encoded_image}})

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_KEY}"
        resp = requests.post(url, json={"contents": [{"parts": parts}]}, headers={'Content-Type': 'application/json'})
        if resp.status_code == 200:
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
        else: return "⚠️ AI Error"
    except Exception: return "⚠️ Connection Failed"

def send_telegram(price, trend, analysis, chart_file, alert_reason):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    text = f"🚨 **WTI ALERT: {alert_reason}** 🚨\nPrice: ${price:.2f} | Trend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- PARALLEL PROCESS: THE BOT LOOP ---
def run_bot():
    global DASHBOARD_DATA
    last_full_report_time, last_price = 0, 0
    seen_news_links = set()

    while True:
        try:
            DASHBOARD_DATA["status"] = "Fetching Market Data..."
            data, price, rsi, trend, atr, ema50, ema21, recent_low, plot_data = get_market_data()
            headlines, raw_entries = get_news()
            
            if data is None:
                time.sleep(120)
                continue

            # Update Raw Technicals HUD
            DASHBOARD_DATA["price"] = price
            DASHBOARD_DATA["trend"] = trend
            DASHBOARD_DATA["news"] = headlines
            DASHBOARD_DATA["rsi"] = rsi
            DASHBOARD_DATA["atr"] = atr
            DASHBOARD_DATA["ema_status"] = "BULLISH (21 > 50)" if ema21 > ema50 else "BEARISH (21 < 50)"
            DASHBOARD_DATA["trade_entry"] = price
            
            current_time = time.time()
            DASHBOARD_DATA["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

            # Fast Spike Checker
            if last_price > 0 and abs(price - last_price) >= 0.50:
                diff = price - last_price
                direction = "UPWARDS 🟢" if diff > 0 else "DOWNWARDS 🔴"
                spike_msg = f"⚡ **FAST SPIKE ALERT** ⚡\nMoved **${abs(diff):.2f} {direction}**!\nPrice: ${price:.2f}"
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': spike_msg})
                last_price = price
                time.sleep(120)
                continue 

            is_emergency = False
            alert_reason = "Regular Check"

            current_utc = calendar.timegm(time.gmtime())
            for entry in raw_entries:
                if 'published_parsed' in entry:
                    age_in_seconds = current_utc - calendar.timegm(entry.published_parsed)
                    if age_in_seconds < 900 and entry.link not in seen_news_links:
                        is_emergency = True
                        alert_reason = f"BREAKING: {entry.title.split(' - ')[0]}"
                        seen_news_links.add(entry.link)
                        break 

            time_since_last_report = current_time - last_full_report_time
            is_time_up = time_since_last_report >= 1800 

            if is_emergency or is_time_up:
                DASHBOARD_DATA["status"] = "Gemini Analyzing Chart..."
                chart = create_chart(plot_data)
                analysis = analyze_market(price, rsi, trend, atr, ema50, ema21, recent_low, headlines, alert_reason, chart)
                
                # Format text for web
                DASHBOARD_DATA["analysis"] = analysis.replace('\n', '<br>')
                
                # --- DATA EXTRACTION MAGIC (Parsing Gemini's brain for the UI) ---
                action_match = re.search(r'Action:\s*(BUY|SELL|STRICT WAIT)', analysis, re.IGNORECASE)
                DASHBOARD_DATA["trade_action"] = action_match.group(1).upper() if action_match else "STRICT WAIT"
                
                # UPDATED BULLETPROOF REGEX FOR CONVICTION & RISK
                conv_match = re.search(r'CONVICTION SCORE[^\d]*(\d+%)', analysis, re.IGNORECASE)
                DASHBOARD_DATA["conviction"] = conv_match.group(1) if conv_match else "0%"
                
                risk_match = re.search(r'RISK LEVEL[^\w]*(.*?)(?:<br>|\n)', DASHBOARD_DATA["analysis"])
                DASHBOARD_DATA["risk"] = risk_match.group(1).replace('*', '').strip() if risk_match else "UNKNOWN"
                
                stop_match = re.search(r'Stop:\s*\$?([0-9.]+)', analysis)
                DASHBOARD_DATA["trade_stop"] = float(stop_match.group(1)) if stop_match else 0.00
                
                target_match = re.search(r'Target:\s*\$?([0-9.]+)', analysis)
                DASHBOARD_DATA["trade_target"] = float(target_match.group(1)) if target_match else 0.00
                # -----------------------------------------------------------------

                if not (MUTE_WAIT_SIGNALS and "STRICT WAIT" in analysis.upper() and not is_emergency):
                    send_telegram(price, trend, analysis, chart, alert_reason)
                
                last_full_report_time = time.time()
                last_price = price
            
            DASHBOARD_DATA["status"] = "Monitoring Active"
                
        except Exception as e:
            DASHBOARD_DATA["status"] = f"Error: {e}"
            print(f"Error: {e}")
        
        time.sleep(120)

# --- WEB UI (TAILWIND CSS DASHBOARD) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WTI QUANT TERMINAL</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { background-color: #0b0f19; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
        .glass-panel { background: rgba(15, 23, 42, 0.8); border: 1px solid #1e293b; border-radius: 8px; }
        .neon-red { color: #f87171; text-shadow: 0 0 10px rgba(248,113,113,0.5); }
        .neon-green { color: #4ade80; text-shadow: 0 0 10px rgba(74,222,128,0.5); }
        .neon-blue { color: #60a5fa; text-shadow: 0 0 10px rgba(96,165,250,0.5); }
        .blink { animation: blinker 1.5s linear infinite; }
        @keyframes blinker { 50% { opacity: 0; } }
    </style>
</head>
<body class="p-6">
    <div class="max-w-7xl mx-auto space-y-6">
        
        <header class="flex justify-between items-center pb-4 border-b border-slate-800">
            <h1 class="text-2xl font-bold tracking-widest text-slate-200">WTI CRUDE OIL <span class="text-xs text-green-400 align-top blink">● LIVE</span></h1>
            <div class="text-sm text-slate-400">Status: <span class="text-yellow-400">{{ data.status }}</span> | Last Update: {{ data.last_update }}</div>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
            
            <div class="glass-panel p-6 flex flex-col justify-center items-center">
                <p class="text-slate-400 text-sm uppercase tracking-wider mb-2">Spot Price</p>
                <h2 class="text-5xl font-bold {% if 'BULLISH' in data.trend %}neon-green{% else %}neon-red{% endif %}">${{ "%.2f"|format(data.price) }}</h2>
                <p class="mt-2 text-xs text-slate-500">Trend: {{ data.trend }}</p>
            </div>
            
            <div class="glass-panel p-6 relative overflow-hidden md:col-span-2">
                <div class="absolute top-0 left-0 w-full h-1 {% if 'BUY' in data.trade_action %}bg-green-500{% elif 'SELL' in data.trade_action %}bg-red-500{% else %}bg-slate-500{% endif %}"></div>
                <p class="text-slate-400 text-sm uppercase tracking-wider mb-4 border-b border-slate-700 pb-1">Live Trade Setup</p>
                
                <div class="flex justify-between items-center mb-4">
                    <div>
                        <p class="text-xs text-slate-500">ACTION</p>
                        <p class="text-3xl font-bold tracking-widest {% if 'BUY' in data.trade_action %}neon-green{% elif 'SELL' in data.trade_action %}neon-red{% else %}text-slate-300{% endif %}">{{ data.trade_action }}</p>
                    </div>
                    <div class="text-right">
                        <p class="text-xs text-slate-500">CONVICTION</p>
                        <p class="text-2xl font-bold text-yellow-400">{{ data.conviction }}</p>
                    </div>
                </div>
                
                <div class="grid grid-cols-3 gap-4 text-center border-t border-slate-700 pt-3">
                    <div>
                        <p class="text-xs text-slate-500">ENTRY</p>
                        <p class="text-slate-200 font-mono text-lg">${{ "%.2f"|format(data.trade_entry) }}</p>
                    </div>
                    <div>
                        <p class="text-xs text-slate-500">TARGET (TP)</p>
                        <p class="text-green-400 font-mono text-lg">${{ "%.2f"|format(data.trade_target) }}</p>
                    </div>
                    <div>
                        <p class="text-xs text-slate-500">STOP LOSS (SL)</p>
                        <p class="text-red-400 font-mono text-lg">${{ "%.2f"|format(data.trade_stop) }}</p>
                    </div>
                </div>
            </div>

            <div class="glass-panel p-6 flex flex-col justify-between">
                <p class="text-slate-400 text-sm uppercase tracking-wider mb-2 border-b border-slate-700 pb-1">Technicals HUD</p>
                <div class="flex justify-between items-center mt-2">
                    <span class="text-slate-400 text-sm">RSI (14)</span>
                    <span class="font-mono font-bold {% if data.rsi > 70 %}text-red-400{% elif data.rsi < 30 %}text-green-400{% else %}text-slate-200{% endif %}">{{ "%.2f"|format(data.rsi) }}</span>
                </div>
                <div class="flex justify-between items-center mt-3">
                    <span class="text-slate-400 text-sm">Vol (ATR)</span>
                    <span class="font-mono text-indigo-400">${{ "%.2f"|format(data.atr) }}</span>
                </div>
                <div class="flex justify-between items-center mt-3">
                    <span class="text-slate-400 text-sm">EMA Trend</span>
                    <span class="font-mono text-xs {% if 'BULLISH' in data.ema_status %}text-green-400{% else %}text-red-400{% endif %}">{{ data.ema_status }}</span>
                </div>
                <div class="flex justify-between items-center mt-3 border-t border-slate-700 pt-2">
                    <span class="text-slate-400 text-xs text-left">{{ data.risk }}</span>
                </div>
            </div>
            
        </div>

        <div class="glass-panel p-4 text-sm">
             <p class="text-slate-400 uppercase tracking-wider mb-2 border-b border-slate-700 pb-1">AI Reasoning Log</p>
             <div class="text-slate-300 leading-relaxed font-sans">{{ data.analysis | safe }}</div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="glass-panel p-6 lg:col-span-2">
                <p class="text-slate-400 text-sm uppercase tracking-wider mb-4 border-b border-slate-700 pb-1">Dual-Channel Trajectory (1H)</p>
                <img src="/chart" alt="WTI Chart" class="w-full h-auto rounded border border-slate-700 shadow-2xl">
            </div>

            <div class="glass-panel p-6 overflow-y-auto max-h-[500px]">
                <p class="text-slate-400 text-sm uppercase tracking-wider mb-4 border-b border-slate-700 pb-1">48HR Macro Timeline</p>
                <div class="space-y-4">
                    {% for item in data.news %}
                    <div class="border-l-2 border-indigo-500 pl-3">
                        <p class="text-xs text-indigo-400 mb-1">{{ item.time }}</p>
                        <a href="{{ item.link }}" target="_blank" class="text-sm text-slate-200 hover:text-white transition-colors block">{{ item.title }}</a>
                    </div>
                    {% endfor %}
                    {% if not data.news %}
                        <p class="text-slate-500 text-sm">Fetching intel...</p>
                    {% endif %}
                </div>
            </div>
        </div>
        
    </div>
    <script>
        // Auto-refresh the page every 60 seconds
        setTimeout(function(){ location.reload(); }, 60000);
    </script>
</body>
</html>
"""

# --- WEB ROUTES ---
@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE, data=DASHBOARD_DATA)

@app.route('/chart')
def serve_chart():
    if os.path.exists("oil_chart.png"):
        return send_file("oil_chart.png", mimetype='image/png')
    return "Chart generating, please refresh in a moment...", 404

# --- RUN EVERYTHING ---
if __name__ == "__main__":
    print("🚀 Firing up Parallel Engines: Bot + Dashboard Server...")
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
