import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import re

# Load your pre-trained model and vectorizer
model = load_model('your_model_path.h5')  # replace with your model file path
tfidf_vectorizer = joblib.load('your_vectorizer_path.pkl')  # replace with your vectorizer file path

# Streamlit application
st.title("Sentiment Analysis of Tweets")

# Function to clean and preprocess tweet
def preprocess_tweet(tweet):
    tweet = re.sub(r"@[A-Za-z0-9]+", "", tweet)  # Remove @mentions
    tweet = re.sub(r"[^a-zA-Z#]", " ", tweet)  # Remove special characters
    tweet = ' '.join([word for word in tweet.split() if len(word) > 3])  # Remove short words
    return tweet

# Input for user to enter a tweet
user_input = st.text_area("Enter your tweet:")

if st.button("Analyze"):
    if user_input:
        # Preprocess the tweet
        cleaned_tweet = preprocess_tweet(user_input)
        
        # Transform using TF-IDF vectorizer
        tweet_vector = tfidf_vectorizer.transform([cleaned_tweet]).toarray()
        
        # Make prediction
        prediction = model.predict(tweet_vector)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        
        # Display the result
        st.write(f"The sentiment of the tweet is: **{sentiment}**")
    else:
        st.write("Please enter a tweet to analyze.")
