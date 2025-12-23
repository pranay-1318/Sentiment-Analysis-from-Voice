import streamlit as st
import speech_recognition as sr
import nltk
import re
import pickle
import tempfile
import numpy as np
from torch import softmax

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load Model & Vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Speech to Text
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return text

# for confidence score
def softmax(x):
    exp_x = np.exp(x - np.max(x)) 
    return exp_x / exp_x.sum()

# Streamlit UI
st.set_page_config(page_title="Voice Emotion Detection", layout="centered")
st.title("🎙️ Voice-based Emotion Detection")

option = st.radio(
    "Choose input method:",
    ("Upload Audio File", "Use Microphone")
)

# OPTION 1: Upload Audio
if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file)

        try:
            text = speech_to_text(uploaded_file)
            st.subheader("📝 Recognized Text")
            st.info(f"_**{text}**_")

            processed_text = preprocess_text(text)
            vector = vectorizer.transform([processed_text])
            # st.write(vector.shape)
            
            scores = model.decision_function(vector)
            # st.write(scores.shape)
            probs = softmax(scores[0])

            emotion = model.predict(vector)[0]
            confidence = np.max(probs) * 100

            st.subheader("🎯 Sentiment Result")
            if emotion == "happy":
                st.success("**😊 Happy**")
            elif emotion == "sad":
                st.warning("**😢 Sad**")
            elif emotion == "angry":
                st.error("**😡 Angry**")
            else:
                st.info("**😐 Neutral**")

            st.subheader("📊 Confidence Score")
            st.warning(f"**{confidence:.2f}%**")  # :.2f for two decimal
            
        except Exception:
            st.error("Could not process the audio. Please upload a clear WAV file.")

# OPTION 2: Microphone Input
if option == "Use Microphone":
    audio_bytes = st.audio_input("🎤 Speak now")

    if audio_bytes:
        # st.audio(audio_bytes)

        # Save audio to temp WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes.read())
            audio_path = f.name

        try:
            text = speech_to_text(audio_path)
            st.subheader("📝 Recognized Text")
            st.info(f"_**{text}**_")

            processed_text = preprocess_text(text)
            vector = vectorizer.transform([processed_text])
            # st.write(vector.shape)
            
            scores = model.decision_function(vector)
            # st.write(scores.shape)
            probs = softmax(scores[0])

            emotion = model.predict(vector)[0]
            confidence = np.max(probs) * 100

            st.subheader("🎯 Sentiment Result")
            if emotion == "happy":
                st.success("**😊 Happy**")
            elif emotion == "sad":
                st.warning("**😢 Sad**")
            elif emotion == "angry":
                st.error("**😡 Angry**")
            else:
                st.info("**😐 Neutral**")

            st.subheader("📊 Confidence Score")
            st.warning(f"**{confidence:.2f}%**")  # :.2f for two decimal

        except Exception as e:
            st.error("❌ System not recognize speech")

