import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("?? Fake News Detection App")

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf-vectorizer.pkl", "rb"))

user_input = st.text_area("Enter a news headline or text:")

if st.button("Check"):
    input_data = vectorizer.transform([user_input])
    result = model.predict(input_data)[0]
    if result == 0:
        st.success("? This news seems **REAL**.")
    else:
        st.error("?? This news seems **FAKE**.")
