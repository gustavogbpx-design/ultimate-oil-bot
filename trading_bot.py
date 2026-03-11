import os
import time
import calendar
import base64 
import threading # NEW: For running bot and web server at the same time
import yfinance as yf
import requests
import feedparser
import pandas as pd
import numpy as np
import mplfinance as mpf
from flask import Flask, render_template_string, send_file # NEW: Web Server
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# --- 1. SETUP KEYS & SETTINGS ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MUTE_WAIT_SIGNALS = True 

# --- GLOBAL STATE FOR THE DASHBOARD ---
# The bot will update this dictionary, and the website will read from it.
DASHBOARD_DATA = {
    "price": 0.00,
    "trend": "BOOTING...",
    "analysis": "Awaiting initial quant calculation...",
    "news": [],
    "last_update": "N/A",
    "status": "Scanning..."
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
    # Changed chart background to deep dark blue to match your requested UI
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
            headlines.append({"title": title, "time": pub_time, "link": entry.link})
            raw_entries.append(entry)
            
        return headlines, raw_entries
    except:
        return [], []

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

def analyze_market(price, rsi, trend, atr, ema50, ema21, recent_low, headlines, alert_reason, chart_file):
    model_name = get_valid_model()
    news_text = "\n".join([f"- {h['title']}" for h in headlines])
    current_time_str = time.strftime("%A, %b %d, %Y - %H:%M UTC", time.gmtime())
    
    stop_loss_buy = price - (1.5 * atr)
    take_profit_buy = price + (2.5 * atr)
    stop_loss_sell = price + (1.5 * atr) 
    take_profit_sell = price - (2.5 * atr)
    ema_status = "BULLISH" if ema21 > ema50 else "BEARISH"

    prompt = f"""
    Act as a Tactical Hedge Fund Algo (Day Trading Desk). 
    🚨 SYSTEM ALERT TRIGGER: {alert_reason} 🚨
    ⏰ CURRENT SYSTEM TIME: {current_time_str}
    
    GLOBAL NEWS FEED:
    {news_text}
    
    TECHNICAL DATA (1-Hour Chart - WTI Crude Oil):
    - Price: ${price:.2f}
    - Absolute Hard Floor Support: ${recent_low:.2f}
    - EMA Trend: {ema_status}
    - RSI: {rsi:.2f}
    - Volatility (ATR): {atr:.2f}
    
    IMAGE ANALYSIS DIRECTIVE: Look at the attached image. Where is the price touching the lines?
    
    TASK: Provide a highly concise quant summary.
    OUTPUT FORMAT:
    🎯 **CONVICTION SCORE: [0-100]%**
    ⚠️ **RISK LEVEL: [🟢 LOW / 🟡 MEDIUM / 🔴 HIGH]**
    💎 **TRADE DECISION**: [BUY / SELL / STRICT WAIT] at ${price:.2f}
    📊 **REASONING**: [1-2 sentences max]
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

            DASHBOARD_DATA["price"] = price
            DASHBOARD_DATA["trend"] = trend
            DASHBOARD_DATA["news"] = headlines
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
                
                DASHBOARD_DATA["analysis"] = analysis.replace('\n', '<br>')
                
                if not (MUTE_WAIT_SIGNALS and "STRICT WAIT" in analysis.upper() and not is_emergency):
                    send_telegram(price, trend, analysis, chart, alert_reason)
                
                last_full_report_time = time.time()
                last_price = price
            
            DASHBOARD_DATA["status"] = "Monitoring Active"
                
        except Exception as e:
            DASHBOARD_DATA["status"] = f"Error: {e}"
        
        time.sleep(120)

# --- WEB UI (TAILWIND CSS DASHBOARD) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLOBAL COMMODITIES • LIVE</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { background-color: #0b0f19; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; }
        .glass-panel { background: rgba(15, 23, 42, 0.8); border: 1px solid #1e293b; border-radius: 8px; }
        .neon-red { color: #f87171; text-shadow: 0 0 10px rgba(248,113,113,0.5); }
        .neon-green { color: #4ade80; text-shadow: 0 0 10px rgba(74,222,128,0.5); }
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

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="glass-panel p-6 flex flex-col justify-center items-center">
                <p class="text-slate-400 text-sm uppercase tracking-wider mb-2">Current Spot Price</p>
                <h2 class="text-6xl font-bold {% if 'BULLISH' in data.trend %}neon-green{% else %}neon-red{% endif %}">${{ "%.2f"|format(data.price) }}</h2>
                <p class="mt-2 text-sm text-slate-500">Trend: {{ data.trend }}</p>
            </div>
            
            <div class="glass-panel p-6 md:col-span-2 overflow-y-auto max-h-48 text-sm">
                <p class="text-slate-400 uppercase tracking-wider mb-2 border-b border-slate-700 pb-1">Gemini Quant Assessment</p>
                <div class="text-slate-300 leading-relaxed">{{ data.analysis | safe }}</div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="glass-panel p-6 lg:col-span-2">
                <p class="text-slate-400 text-sm uppercase tracking-wider mb-4 border-b border-slate-700 pb-1">Dual-Channel Trajectory (1H)</p>
                <img src="/chart" alt="WTI Chart" class="w-full h-auto rounded border border-slate-700 shadow-2xl">
            </div>

            <div class="glass-panel p-6 overflow-y-auto max-h-[600px]">
                <p class="text-slate-400 text-sm uppercase tracking-wider mb-4 border-b border-slate-700 pb-1">Macro Event Timeline</p>
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
        // Auto-refresh the page every 60 seconds to get new data
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
    # Serves the actual image generated by the bot
    if os.path.exists("oil_chart.png"):
        return send_file("oil_chart.png", mimetype='image/png')
    return "Chart generating, please refresh in a moment...", 404

# --- RUN EVERYTHING ---
if __name__ == "__main__":
    print("🚀 Firing up Parallel Engines: Bot + Dashboard Server...")
    
    # 1. Start the Telegram/Gemini Trading Bot in a separate background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # 2. Start the Flask Web Server on the main thread
    # Note: Railway requires host='0.0.0.0' and uses the PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
