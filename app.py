import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# ------------------ DISCLAIMER SECTION ------------------
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    st.title("⚠️ Disclaimer (Legal Notice)")
    st.markdown("""
    **यह ऐप केवल शैक्षिक (educational) और डेमो उद्देश्य के लिए बनाया गया है।**

    - यह किसी प्रकार की निवेश सलाह (investment advice) नहीं है।
    - इसमें दिखाए गए प्रेडिक्शन या संकेत मात्र एक तकनीकी अभ्यास का हिस्सा हैं।
    - इस ऐप का प्रयोग वित्तीय जोखिम (financial risk) से जुड़ा हो सकता है।
    - भारतीय कानून (SEBI Act, 1992 & Securities Contracts Regulation Act, 1956) के अंतर्गत हम निवेश सलाह देने के पात्र नहीं हैं।
    - हम किसी भी प्रकार के नुकसान या हानि (losses) के लिए ज़िम्मेदार नहीं हैं।
    - इसका बार-बार उपयोग करने की आदत (addiction) लग सकती है। कृपया विवेकपूर्ण निर्णय लें।

    **By clicking the button below, you accept these terms and understand all risks.**
    """)

    if st.button("✅ I Understand and Accept"):
        st.session_state.disclaimer_accepted = True
    st.stop()

# ------------------ APP STARTS HERE ------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Predictor with Technicals + News Sentiment")

def fetch_data(symbol):
    return yf.download(symbol, period="6mo", interval="1d")

def add_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['STD'] = df['Close'].rolling(20).std()
    df['Upper'] = df['SMA_20'] + 2 * df['STD']
    df['Lower'] = df['SMA_20'] - 2 * df['STD']
    df['MACD'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()
    df['Signal'] = df['MACD'].ewm(9).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + avg_gain/avg_loss))
    return df.dropna()

def fetch_sentiment(symbol):
    url = f"https://www.google.com/search?q={symbol}+stock+news"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    snippets = soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")
    sentiments = []
    for tag in snippets:
        text = tag.get_text()
        blob = TextBlob(text)
        sentiments.append(blob.sentiment.polarity)
    return round(np.mean(sentiments), 3) if sentiments else 0

def predict(df):
    df = df.reset_index(drop=True)
    df['Idx'] = df.index.values
    X = df[['Idx']]
    y = df[['Close']]
    model = LinearRegression().fit(X, y)
    next_idx = np.array([[df['Idx'].iloc[-1] + 1]])
    fifth_idx = np.array([[df['Idx'].iloc[-1] + 5]])
    npred = float(model.predict(next_idx))
    fpred = float(model.predict(fifth_idx))
    direction = "UP" if npred > df['Close'].iloc[-1] else "DOWN"
    acc = round(model.score(X, y) * 100, 2)
    return npred, fpred, direction, acc

symbol = st.text_input("Enter Stock Symbol (e.g. TATAPOWER.NS):", "TATAPOWER.NS")

if st.button("📊 Predict"):
    df = fetch_data(symbol)
    if df.empty:
        st.error("❌ Invalid symbol or no data.")
    else:
        df = add_indicators(df)
        sentiment = fetch_sentiment(symbol)
        npred, fpred, direction, acc = predict(df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Next‑Day Target", f"₹{npred:.2f}")
            st.metric("5th‑Day Target", f"₹{fpred:.2f}")
            st.metric("Direction", direction)
            st.metric("Accuracy", f"{acc}%")
            st.metric("News Sentiment", sentiment)

        with col2:
            st.subheader("📉 Chart")
            st.line_chart(df[['Close','SMA_20','EMA_20','Upper','Lower']])
            st.subheader("📄 Last 5 Rows")
            st.dataframe(df.tail(5))
