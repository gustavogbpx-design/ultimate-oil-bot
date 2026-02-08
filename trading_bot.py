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

# Global History for volatility tracking
price_history = [] 

# --- DATA FETCHING ---
def get_market_data():
    ticker = "CL=F" # WTI Crude Oil
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
        print(f"Data Error: {e}")
        return None, 0, 0, "Error", 0, 0

# --- CHART GENERATION ---
def create_chart(data):
    if data is None: return None
    fname = "oil_chart.png"
    # Adds 50 and 200 Moving Averages to the chart visual
    mpf.plot(data.tail(50), type='candle', style='charles', volume=False, 
             mav=(50, 200), savefig=fname)
    return fname

# --- NEWS FETCHING ---
def get_news():
    try:
        feed = feedparser.parse("https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Iran+Conflict&hl=en-US&gl=US&ceid=US:en")
        if not feed.entries: return []
        return [entry.title for entry in feed.entries[:5]]
    except:
        return []

# --- AI ANALYSIS ---
def analyze_market(price, rsi, trend, atr, ema200, headlines):
    
    # --- MATH: Dynamic Stop Loss Calculation ---
    # Volatility Adjustment: Wider stops when ATR is high
    stop_loss_buy = price - (2.0 * atr)
    take_profit_buy = price + (3.0 * atr) # 1.5 Risk/Reward Ratio
    
    stop_loss_sell = price + (2.0 * atr)
    take_profit_sell = price - (3.0 * atr)
    
    # Trend Filter Status
    ema_status = "ABOVE 200 EMA (Uptrend)" if price > ema200 else "BELOW 200 EMA (Downtrend)"

    news_text = "\n".join([f"- {h}" for h in headlines])
    
    # PROMPT
    prompt = f"""
    Act as a Senior Risk Manager at a Hedge Fund.
    
    MARKET DATA (WTI Oil):
    - Price: ${price:.2f}
    - RSI: {rsi:.2f} (Overbought > 70, Oversold < 30)
    - Trend: {trend}
    - EMA Context: {ema_status}
    - Volatility (ATR): {atr:.2f}
    
    CALCULATED RISK LIMITS (Based on ATR):
    - Suggested BUY Stop Loss: ${stop_loss_buy:.2f}
    - Suggested SELL Stop Loss: ${stop_loss_sell:.2f}
    
    NEWS HEADLINES:
    {news_text}
    
    TASK:
    Analyze the setup. Recommend a trade only if risk is manageable.
    
    OUTPUT FORMAT:
    
    💎 **TRADE SIGNAL**
    Action: [BUY / SELL / WAIT]
    Entry: ${price:.2f}
    🛡️ Smart Stop Loss: [Use calculated value]
    🎯 Target: [Use calculated value]
    
    📊 **RISK ASSESSMENT**
    Risk Level: [Low/Med/High]
    Reasoning:
    - [Technical Analysis]
    - [News Sentiment]
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
def send_telegram(price, analysis, chart_file, alert_reason):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})
    
    text = f"🚨 **MARKET ALERT ({alert_reason})**\nPrice: ${price:.2f}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("🚀 Professional Trading Bot Started (ATR + 200 EMA Mode)...")
    
    while True:
        try:
            print("Analyzing market structure...")
            data, price, rsi, trend, atr, ema200 = get_market_data()
            
            if data is not None:
                # 1. Update Price History for Volatility Check
                current_time = time.time()
                price_history.append((current_time, price))
                price_history = [p for p in price_history if current_time - p[0] <= 3600]
                
                # 2. Check for Sudden Price Shock ($1 Move in 1 Hour)
                start_price = price_history[0][1]
                price_change = abs(price - start_price)
                is_volatile = price_change >= 1.0

                # 3. Analyze
                headlines = get_news()
                chart = create_chart(data)
                analysis = analyze_market(price, rsi, trend, atr, ema200, headlines)

                # 4. Filter: Only Send if Low/Med Risk OR High Volatility
                is_safe = "Risk Level: Low" in analysis or "Risk Level: Medium" in analysis
                
                if is_safe:
                    print("✅ High Quality Setup Detected. Sending Report...")
                    send_telegram(price, analysis, chart, "Trade Opportunity")
                elif is_volatile:
                    print(f"⚠️ Market Shock Detected! Price moved ${price_change:.2f}")
                    send_telegram(price, analysis, chart, "Volatility Alert")
                else:
                    print("⏳ Market is choppy/risky. Waiting for better setup.")

            else:
                print("❌ Data Feed Error.")

        except Exception as e:
            print(f"⚠️ Runtime Error: {e}")
        
        # Wait 10 Minutes
        print("⏳ Next scan in 10 minutes...")
        time.sleep(600)
