import os
import yfinance as yf
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

# --- 2. GET DATA ---
def get_market_data():
    ticker = "CL=F"
    try:
        data = yf.download(ticker, period="5d", interval="30m", progress=False)
        if data.empty: return None, 0, 0, "No Data"
        
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)
        close = data["Close"]
        if hasattr(close, "shape") and len(close.shape) > 1: close = close.iloc[:, 0]
        data["Close"] = close

        data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
        macd = MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        
        price = data["Close"].iloc[-1]
        rsi = data["RSI"].iloc[-1]
        trend = "BULLISH 🟢" if data["MACD"].iloc[-1] > data["Signal"].iloc[-1] else "BEARISH 🔴"
        
        return data, price, rsi, trend
    except Exception as e:
        print(f"Data Error: {e}")
        return None, 0, 0, "Error"

# --- 3. DRAW CHART ---
def create_chart(data):
    if data is None: return None
    fname = "oil_chart.png"
    mpf.plot(data.tail(40), type='candle', style='charles', volume=False, savefig=fname)
    return fname

# --- 4. GET NEWS HEADLINES ---
def get_news():
    try:
        # We search specifically for Oil, War, and OPEC news
        feed = feedparser.parse("https://news.google.com/rss/search?q=Crude+Oil+OR+OPEC+OR+Iran+Conflict&hl=en-US&gl=US&ceid=US:en")
        if not feed.entries: return []
        # Return top 5 headlines
        return [entry.title for entry in feed.entries[:5]]
    except:
        return []

# --- 5. HYBRID ANALYSIS ENGINE ---
def analyze_market(price, rsi, trend, headlines):
    
    # STEP A: Try Google AI (The Smart Brain)
    try:
        news_text = "\n".join([f"- {h}" for h in headlines])
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
        prompt = f"""
        You are a hedge fund trader. 
        Market Data: Price ${price:.2f}, RSI {rsi:.2f}, Trend {trend}.
        News Headlines:
        {news_text}
        
        Task: DECIDE to BUY or SELL. 
        Logic: If news is about War/Attacks/OPEC Cuts -> BUY. If news is Peace/Economy bad -> SELL.
        Output: A short, aggressive trading signal with emojis.
        """
        resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={'Content-Type': 'application/json'})
        
        if resp.status_code == 200:
            return "🧠 **AI BRAIN ANALYSIS:**\n" + resp.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        pass # If AI fails, we fall back to Step B silently
    
    # STEP B: The "News Sentiment Algo" (Backup Brain)
    # This runs if Google AI is broken. It calculates a score based on words.
    
    score = 0
    bullish_words = ["war", "attack", "conflict", "strike", "escalat", "cut", "opec", "shortage", "sanction", "tension"]
    bearish_words = ["peace", "talks", "ceasefire", "deal", "supply", "rise", "increase", "recession", "weak", "surplus"]
    
    matches = []
    
    for headline in headlines:
        lower_h = headline.lower()
        # Check Bullish
        for word in bullish_words:
            if word in lower_h:
                score += 1
                matches.append(f"🔥 Bullish News: {word.upper()} found")
                break
        # Check Bearish
        for word in bearish_words:
            if word in lower_h:
                score -= 1
                matches.append(f"❄️ Bearish News: {word.upper()} found")
                break
                
    # Final Decision Logic
    signal = "WAIT ✋"
    if score > 0: signal = "BUY 🟢 (War/Supply Risks)"
    if score < 0: signal = "SELL 🔴 (Peace/Supply Glut)"
    
    # RSI Filters
    if rsi < 30: signal = "BUY 🟢 (Oversold Bounce)"
    if rsi > 70: signal = "SELL 🔴 (Overbought Drop)"

    analysis_text = f"⚙️ **ALGO ANALYSIS (Backup):**\nSignal: **{signal}**\nNews Score: {score}\n\n"
    if matches:
        analysis_text += "📝 **Key News Factors:**\n" + "\n".join(matches)
    else:
        analysis_text += "No major war/peace keywords found in news."
        
    return analysis_text

# --- 6. SEND TELEGRAM ---
def send_telegram(price, rsi, trend, analysis, chart_file):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    
    # Send Chart
    if chart_file:
        with open(chart_file, 'rb') as f:
            requests.post(f"{base_url}/sendPhoto", data={'chat_id': TELEGRAM_CHAT_ID}, files={'photo': f})

    # Send Analysis
    text = f"🛢 **WTI INTELLIGENCE**\nPrice: ${price:.2f}\nRSI: {rsi:.2f}\nTrend: {trend}\n\n{analysis}"
    requests.post(f"{base_url}/sendMessage", data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

# --- MAIN ---
if __name__ == "__main__":
    data, price, rsi, trend = get_market_data()
    if data is not None:
        chart = create_chart(data)
        headlines = get_news()
        analysis = analyze_market(price, rsi, trend, headlines)
        send_telegram(price, rsi, trend, analysis, chart)
        print("Sent successfully.")
