import os
import yfinance as yf
import google.generativeai as genai
import requests
import feedparser
import pandas as pd
import mplfinance as mpf
from ta.momentum import RSIIndicator
from ta.trend import MACD

# --- 1. SETUP KEYS ---
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

genai.configure(api_key=GEMINI_KEY)

# --- 2. GET DATA & CALCULATE INDICATORS ---
def get_market_data():
    ticker = "CL=F"  # WTI Crude Oil
    # Download 5 days of data to make the chart look nice
    data = yf.download(ticker, period="5d", interval="30m", progress=False)
    
    # Calculate RSI (The Bounce Indicator)
    rsi_ind = RSIIndicator(close=data["Close"], window=14)
    data["RSI"] = rsi_ind.rsi()
    
    # Calculate MACD (The Trend Indicator)
    macd_ind = MACD(close=data["Close"])
    data["MACD"] = macd_ind.macd()
    data["Signal"] = macd_ind.macd_signal()
    
    # Get latest values
    last_price = data["Close"].iloc[-1]
    last_rsi = data["RSI"].iloc[-1]
    last_macd = data["MACD"].iloc[-1]
    last_signal = data["Signal"].iloc[-1]
    
    # Determine Trend
    trend = "BULLISH (Up)" if last_macd > last_signal else "BEARISH (Down)"
    
    return data, last_price, last_rsi, trend

# --- 3. DRAW THE CHART ---
def create_chart_image(data):
    filename = "oil_chart.png"
    
    # Style the chart (Green up, Red down)
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc, style='yahoo')
    
    # Draw only the last 40 candles so it's zoomed in
    subset = data.tail(40)
    
    mpf.plot(subset, type='candle', style=style, 
             title="WTI Oil (30m)", 
             volume=False, 
             savefig=filename)
    
    return filename

# --- 4. GET NEWS ---
def get_news():
    url = "https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Iran+War&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    headlines = ""
    for i, entry in enumerate(feed.entries[:5]):
        headlines += f"- {entry.title}\n"
    return headlines

# --- 5. ASK GEMINI ---
def ask_gemini(price, rsi, trend, news):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert Oil Trader. Analyze this setup:
    
    DATA:
    - Price: ${price:.2f}
    - RSI: {rsi:.2f} (Over 70=Expensive, Under 30=Cheap)
    - MACD Trend: {trend}
    
    NEWS HEADLINES:
    {news}
    
    TASK:
    Give a trading signal.
    1. If News is scary (War/Supply Cut) -> Bias BUY.
    2. If News is calm/Peace -> Bias SELL.
    3. Use RSI to time the entry.
    
    OUTPUT FORMAT (Telegram Style):
    🔮 **AI PREDICTION**
    **Action:** [BUY / SELL / WAIT]
    **Risk:** [High / Medium / Low]
    **Reason:** [1-2 sentences explaining why]
    """
    response = model.generate_content(prompt)
    return response.text

# --- 6. SEND TO TELEGRAM (PHOTO + TEXT) ---
def send_alert(message, image_file):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    
    # Open the image file we created
    with open(image_file, 'rb') as f:
        files = {'photo': f}
        data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message, 'parse_mode': 'Markdown'}
        requests.post(url, files=files, data=data)

# --- MAIN RUN ---
if __name__ == "__main__":
    try:
        print("1. Fetching Data...")
        data, price, rsi, trend = get_market_data()
        
        print("2. Drawing Chart...")
        chart_file = create_chart_image(data)
        
        print("3. Reading News...")
        news = get_news()
        
        print("4. Asking AI...")
        analysis = ask_gemini(price, rsi, trend, news)
        
        # Build the message
        msg = f"🛢 **WTI LIVE UPDATE** 🛢\nPrice: ${price:.2f}\nRSI: {rsi:.2f}\nTrend: {trend}\n\n{analysis}"
        
        print("5. Sending to Telegram...")
        send_alert(msg, chart_file)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
