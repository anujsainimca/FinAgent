import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import openai
import requests
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import yfinance as yf

# Hardcoded API keys
NEWS_API_KEY = '4b2309aa06a14a8d82e340e72ec8b9a4'


# Function to fetch stock data using yfinance
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    data = yf.download(symbol, period='1y')  # Fetch 1 year of data for the stock symbol
    return data

# Function to fetch news and sentiment
def fetch_news_and_sentiment(company: str):
    url = f"https://newsapi.org/v2/everything?q={company}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    if not articles:
        return f"No news found for {company}", []

    headlines = []
    wordcloud_text = ''
    for article in articles[:5]:
        headline = article['title']
        description = article.get('description', '')
        sentiment = TextBlob(headline + " " + description).sentiment.polarity
        sentiment_type = 'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'
        headlines.append((headline, sentiment_type, sentiment, article['url']))  # Store the URL
        wordcloud_text += f"{headline} {description} "
    
    return headlines, wordcloud_text

# Function for adding custom CSS to apply a background image and style elements
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Background image and styling */
        body {
            background-image: url("https://www.transparenttextures.com/patterns/food.png");
            background-size: cover;
        }
        .title-header {
            font-size: 2.5em;
            color: #333333;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .subheader {
            color: #4CAF50;
            font-size: 1.5em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to apply CSS styles
add_custom_css()

# Add the logo using Streamlit's st.image()
st.image('stock.png', width=100)

# Streamlit UI
st.markdown("<div class='title-header'>Stock Market & Financial News Dashboard</div>", unsafe_allow_html=True)

# Input fields for API key and stock symbol
chatgpt_api_key = st.text_input("Enter your OpenAI API Key", type="password")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)")

# Dropdown to select ChatGPT model
models = ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4']
selected_model = st.selectbox("Choose GPT Model", models, index=0)

# When the user enters both the ChatGPT API key and stock symbol
if chatgpt_api_key and stock_symbol:
    openai.api_key = chatgpt_api_key
    
    # Fetch and display stock data as a table
    st.markdown(f"<div class='subheader'>Stock Data for {stock_symbol}</div>", unsafe_allow_html=True)
    stock_data = fetch_stock_data(stock_symbol)
    st.dataframe(stock_data.head())
    
    # Plot stock data
    st.markdown(f"<div class='subheader'>{stock_symbol} Stock Price Charts</div>", unsafe_allow_html=True)
    fig1 = px.line(stock_data, x=stock_data.index, y='Close', title=f'{stock_symbol} Closing Prices')
    st.plotly_chart(fig1)

    # Moving Averages Chart
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Closing Price'))
    fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='50-day MA'))
    fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], mode='lines', name='200-day MA'))
    fig2.update_layout(title=f'{stock_symbol} Stock Prices with Moving Averages')
    st.plotly_chart(fig2)

    # Additional charts
    st.markdown(f"<div class='subheader'>Additional Stock Charts</div>", unsafe_allow_html=True)
    
    # Volume Chart
    fig3 = px.bar(stock_data, x=stock_data.index, y='Volume', title=f'{stock_symbol} Trading Volume')
    st.plotly_chart(fig3)

    # Candlestick Chart
    fig4 = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                          open=stock_data['Open'],
                                          high=stock_data['High'],
                                          low=stock_data['Low'],
                                          close=stock_data['Close'])])
    fig4.update_layout(title=f'{stock_symbol} Candlestick Chart')
    st.plotly_chart(fig4)

    # Fetch and display financial news with sentiment analysis
    st.markdown(f"<div class='subheader'>Top 5 News Headlines for {stock_symbol}</div>", unsafe_allow_html=True)
    headlines, wordcloud_text = fetch_news_and_sentiment(stock_symbol)

    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    sentiment_colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}

    for idx, (headline, sentiment_type, sentiment, url) in enumerate(headlines):
        color = sentiment_colors[sentiment_type]
        st.markdown(f"<div class='news-headline' style='color: {color};'>{idx+1}. <a href='{url}' target='_blank'>{headline}</a> [Sentiment: {sentiment_type}, Score: {sentiment:.2f}]</div>", unsafe_allow_html=True)
        sentiments[sentiment_type] += 1

    # Display sentiment analysis as a bar chart
    st.markdown(f"<div class='subheader'>Sentiment Distribution for {stock_symbol}</div>", unsafe_allow_html=True)
    fig5 = px.bar(x=list(sentiments.keys()), y=list(sentiments.values()), 
                  labels={'x': 'Sentiment Type', 'y': 'Count'}, title="Sentiment Analysis")
    fig5.update_traces(marker_color=['green' if sentiment == 'Positive' else 'red' if sentiment == 'Negative' else 'gray' for sentiment in sentiments.keys()])
    st.plotly_chart(fig5)

    # Generate WordCloud
    st.markdown("<div class='subheader'>Word Cloud of Top News Concepts</div>", unsafe_allow_html=True)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # OpenAI GPT-3 Prediction
    if st.button("Predict Future Stock Performance Using GPT-3"):
        system_prompt = f"You are a financial Analyst."
        prompt = f"Based on the given financial news and stock data, predict the future performance of {stock_symbol} stock. Give a structured heading and description based response. Here are the news: {str(headlines)}. Also use the latest stock prices to get some predictions. {str(stock_data)}"
        completion = openai.ChatCompletion.create(
            model=selected_model,  # Adjust the model as needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and print the response
        prediction = completion.choices[0].message['content']
        st.markdown(f"<div class='prediction-box'><h3>ChatGPT Prediction for {stock_symbol}</h3><p>{prediction}</p></div>", unsafe_allow_html=True)
