Project Name: Sentiment Analysis from Voice 

Sentiment Analysis from Voice is an AI/ML-based application that converts human speech into text and predicts the underlying sentiment

Setup Step:

1. Dataset Collection & Feature Extraction:
I collected two relevant datasets from Kaggle, extracted the most important features, and merged them into a single refined dataset named final_dataset.csv

2. Data Preprocessing & Vectorization:
The dataset was preprocessed using NLP techniques such as text cleaning, stopword removal, and normalization.
After preprocessing, a TF-IDF vectorization technique was applied to convert textual data into numerical feature vectors.

3. Model Training
A machine learning model was trained for multi-class sentiment classification using LinearSVC.
The trained sentiment classification model and the TF-IDF vectorizer were saved as serialized files:
- emotion_model.pkl
- tfidf_vectorizer.pkl

4. Model Deployment Using Streamlit:
The serialized models were integrated into a Streamlit-based web application (app.py), enabling real-time sentiment analysis from voice input through audio upload and microphone input.

Technologies used in project:
- Python
- SpeechRecognition
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Linear SVM
- Streamlit
- Pickle (.pkl)
  
-------------------------------

User can do:
- Real-time microphone input
- Audio file upload support

Projec do:
- Speech-to-text conversion
- Multi-class sentiment classification then choose one sentiment (happy, sad, angry, neutral)
- Confidence score for predictions
- Deployed using Streamlit
