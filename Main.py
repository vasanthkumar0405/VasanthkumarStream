import streamlit as st
from transformers import pipeline

# Load sentiment analysis pipeline (you can customize this)
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

st.title("Emotion Detection from Social Media Text")
st.write("Enter a sentence below to detect the emotion.")

# Input text
user_input = st.text_area("Enter text here")

# Prediction
if st.button("Analyze Emotion"):
    if user_input.strip():
        prediction = classifier(user_input)[0]
        # Sort by score
        sorted_pred = sorted(prediction, key=lambda x: x['score'], reverse=True)
        top_emotion = sorted_pred[0]
        st.success(f"**Predicted Emotion:** {top_emotion['label']} ({top_emotion['score']:.2f})")
        st.write("**All Scores:**")
        st.json(sorted_pred)
    else:
        st.warning("Please enter some text to analyze.")