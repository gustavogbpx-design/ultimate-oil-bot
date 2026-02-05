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
    data = yf.download(ticker, period="5d", interval="30m", progress=False)
    
    # --- BUG FIX: FLATTEN DATA ---
    if data.empty:
        raise ValueError("No data downloaded!")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    close_prices = data["Close"]
    if hasattr(close_prices, "shape") and len(close_prices.shape) > 1:
        close_prices = close_prices.iloc[:, 0]
    data["Close"] = close_prices

    # Indicators
    rsi_ind = RSIIndicator(close=data["Close"], window=14)
    data["RSI"] = rsi_ind.rsi()
    macd_ind = MACD(close=data["Close"])
    data["MACD"] = macd_ind.macd()
    data["Signal"] = macd_ind.macd_signal()
    
    # Latest Values
    last_price = data["Close"].iloc[-1]
    last_rsi = data["RSI"].iloc[-1]
    last_macd = data["MACD"].iloc[-1]
    last_signal = data["Signal"].iloc[-1]
    trend = "BULLISH (Up)" if last_macd > last_signal else "BEARISH (Down)"
    
    return data, last_price, last_rsi, trend

# --- 3. DRAW THE CHART (SIMPLIFIED) ---
def create_chart_image(data):
    filename = "oil_chart.png"
    subset = data.tail(40)
    
    # We use 'charles' style which is standard Green/Red candles
    mpf.plot(subset, type='candle', style='charles', 
             title="WTI Oil (30m)", 
             volume=False, 
             savefig=filename)
    
    return filename

# --- 4. GET NEWS ---
def get_news():
    try:
        url = "https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Iran+War&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        if not feed.entries: return "No specific news found."
        return "\n".join([f"- {entry.title}" for entry in feed.entries[:5]])
    except:
        return "Could not fetch news."

# --- 5. ASK GEMINI ---
def ask_gemini(price, rsi, trend, news):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are an expert Oil Trader. Analyze this setup:
    DATA: Price ${price:.2f}, RSI {rsi:.2f}, Trend {trend}.
    NEWS: {news}
    TASK: Give a BUY/SELL/WAIT signal based on War Risks + RSI Math.
    OUTPUT: Telegram format with emojis.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI Analysis Unavailable."

# --- 6. SEND TO TELEGRAM ---
def send_alert(message, image_file):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_file, 'rb') as f:
        requests.post(url, files={'photo': f}, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': message, 'parse_mode': 'Markdown'})

# --- MAIN RUN ---
if __name__ == "__main__":
    try:
        print("1. Fetching Data...")
        data, price, rsi, trend = get_market_data()
        print("2. Drawing Chart...")
        chart_file = create_chart_image(data)
        print("3. Reading News & AI...")
        analysis = ask_gemini(price, rsi, trend, get_news())
        print("4. Sending...")
        send_alert(f"🛢 **WTI UPDATE**\nPrice: ${price:.2f}\nRSI: {rsi:.2f}\nTrend: {trend}\n\n{analysis}", chart_file)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
