import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from ntscraper import Nitter

# -------------------------------
# Setup & Caching
# -------------------------------
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return stopwords.words("english")

@st.cache_resource
def load_model_and_vectorizer():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

@st.cache_resource
def initialize_scraper():
    return Nitter(log_level=1)

# -------------------------------
# Helper Functions
# -------------------------------
def preprocess_text(text, stop_words):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

def predict_sentiment(text, model, vectorizer, stop_words):
    processed_text = preprocess_text(text, stop_words)
    features = vectorizer.transform([processed_text])
    sentiment = model.predict(features)[0]
    return "Negative" if sentiment == 0 else "Positive"

def create_card(tweet_text, sentiment):
    color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
    return f"""
    <div style="background-color: {color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h5 style="color: white; margin: 0;">{sentiment} Sentiment</h5>
        <p style="color: white; margin: 5px 0 0 0;">{tweet_text}</p>
    </div>
    """

# -------------------------------
# Main App
# -------------------------------
def main():
    st.title("🐦 Twitter Sentiment Analysis")

    # Load resources
    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    option = st.selectbox("Choose an option", ["Input Text", "Fetch Tweets from User"])

    if option == "Input Text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            if text_input.strip():
                sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
                st.success(f"Sentiment: {sentiment}")
            else:
                st.warning("⚠️ Please enter some text.")

    elif option == "Fetch Tweets from User":
        username = st.text_input("Enter Twitter username (without @)")
        num_tweets = st.slider("Number of tweets to fetch", 1, 20, 5)

        if st.button("Fetch Tweets"):
            if username.strip():
                try:
                    tweets_data = scraper.get_tweets(username, mode="user", number=num_tweets)
                    if "tweets" in tweets_data:
                        for tweet in tweets_data["tweets"]:
                            tweet_text = tweet["text"]
                            sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                            st.markdown(create_card(tweet_text, sentiment), unsafe_allow_html=True)
                    else:
                        st.error("⚠️ No tweets found or an error occurred.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a valid username.")

if __name__ == "__main__":
    main()
