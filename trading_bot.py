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

# --- ASSET LIST ---
# We now watch TWO targets:
ASSETS = {
    "WTI Oil": "CL=F",
    "Gold": "GC=F"
}

# Global History for volatility tracking (Separate memory for each asset)
price_history = {"WTI Oil": [], "Gold": []}

# --- DATA FETCHING ---
def get_market_data(ticker):
    try:
        # Download 5 Days of 30-min candles
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, 0, 0, "No Signal"
        
        # Clean Data Format
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close
        
        # --- TECHNICAL INDICATORS ---
        # 1. RSI (Momentum)
        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        
        # 2. MACD (Trend Momentum)
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        
        # 3. 200 EMA (Long Term Trend Filter)
        data["EMA200"] = EMAIndicator(close=data["Close"], window=200).ema_indicator()
        
        # 4. ATR (Volatility / Risk Measurement)
        data["ATR"] = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14).average_true_range()
        
        # Get latest values
        price = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        atr = data["ATR"].iloc[-1]
        ema200 = data["EMA200"].iloc[-1]
        
        # Determine Trend
        trend = "BULLISH 🟢" if data["MACD"].iloc[-1] > data["Signal"].iloc[-1] else "BEARISH 🔴"
        
        return data, price, rsi, trend, atr, ema200
        
    except Exception as e:
        print(f"Data Error ({ticker}): {e}")
        return None, 0, 0, "Error", 0, 0

# --- CHART GENERATION ---
def create_chart(data, asset_name):
    if data is None: return None
    fname = f"{asset_name.replace(' ', '_')}_chart.png"
    mpf.plot(data.tail(50), type='candle', style='charles', volume=False, 
             mav=(50, 200), title=f"{asset_name} Analysis", savefig=fname)
    return fname

# --- NEWS FETCHING ---
def get_news(asset_name):
    search_term = "Crude+Oil" if "Oil" in asset_name else "Gold+Price"
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={search_term}&hl=en-US&gl=US&ceid=US:en")
        if not feed.entries: return []
        return [entry.title for entry in feed.entries[:3]] # Top 3 headlines
    except:
        return []

# --- AI ANALYSIS ---
def analyze_market(asset, price, rsi, trend, atr, ema200, headlines):
    
    # --- MATH: Dynamic Stop Loss Calculation ---
    stop_loss_buy = price - (2.0 * atr)
    take_profit_buy = price + (3.0 * atr) 
    
    stop_loss_sell = price + (2.0 * atr)
    take_profit_sell = price - (3.0 * atr)
    
    ema_status = "ABOVE 200 EMA (Uptrend)" if price > ema200 else "BELOW 200 EMA (Downtrend)"
    news_text = "\n".join([f"- {h}" for h in headlines])
    
    # PROMPT
    prompt = f"""
    Act as a Senior Hedge Fund Trader.
    
    ASSET: {asset}
    - Price: ${price:.2f}
    - RSI: {rsi:.2f} (Overbought > 70, Oversold < 30)
    - Trend: {trend}
    - EMA Context: {ema_status}
    - Volatility (ATR): {atr:.2f}
    
    RISK LIMITS (ATR Based):
    - Suggested BUY Stop: ${stop_loss_buy:.2f}
    - Suggested SELL Stop: ${stop_loss_sell:.2f}
    
    NEWS:
    {news_text}
    
    TASK:
    Analyze the setup. Is this a high-quality trade?
    
    OUTPUT FORMAT:
    
    💎 **{asset.upper()} SIGNAL**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛡️ Stop Loss: [Calculated Value]
    🎯 Target: [Calculated Value]
    
    📊 **ANALYSIS**
    Risk Level: [Low/Med/High]
    Reasoning: [1-2 sentences on technicals & news]
    """
    
    try:
        model_name = "models/gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={GEMINI_KEY}"
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        if resp.status_code == 200:
            return resp.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        pass
    return "⚠️ AI Analysis Unavailable."

# --- TELEGRAM SENDER ---
def send_telegram(asset, price, analysis, chart_file, alert_reason):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    
    text = f"🚨 **{asset.upper()} ALERT ({alert_reason})**\nPrice: ${price:.2f}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("🚀 Golden Sniper Mode Activated (Oil + Gold)...")
    
    while True:
        try:
            for asset_name, ticker in ASSETS.items():
                print(f"🔍 Scanning {asset_name}...")
                data, price, rsi, trend, atr, ema200 = get_market_data(ticker)
                
                if data is not None:
                    # 1. Update Price History (Specific to this asset)
                    current_time = time.time()
                    price_history[asset_name].append((current_time, price))
                    price_history[asset_name] = [p for p in price_history[asset_name] if current_time - p[0] <= 3600]
                    
                    # 2. Check Volatility (Custom thresholds: Oil $1.00, Gold $10.00)
                    start_price = price_history[asset_name][0][1]
                    price_change = abs(price - start_price)
                    
                    volatility_threshold = 1.0 if asset_name == "WTI Oil" else 10.0
                    is_volatile = price_change >= volatility_threshold

                    # 3. Analyze
                    headlines = get_news(asset_name)
                    chart = create_chart(data, asset_name)
                    analysis = analyze_market(asset_name, price, rsi, trend, atr, ema200, headlines)

                    # 4. Filter
                    is_safe = "Risk Level: Low" in analysis or "Risk Level: Medium" in analysis
                    
                    if is_safe:
                        print(f"✅ {asset_name}: Good Setup.")
                        send_telegram(asset_name, price, analysis, chart, "Opportunity")
                    elif is_volatile:
                        print(f"⚠️ {asset_name}: High Volatility!")
                        send_telegram(asset_name, price, analysis, chart, "Big Move")
                    else:
                        print(f"zzz {asset_name}: No clear trade.")

                time.sleep(5) # Small pause between assets

        except Exception as e:
            print(f"⚠️ Loop Error: {e}")
        
        # Wait 10 Minutes
        print("⏳ Next scan in 10 minutes...")
        time.sleep(600)
