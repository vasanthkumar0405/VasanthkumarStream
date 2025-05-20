import gradio as gr
from transformers import pipeline

# Load a pre-trained emotion classification model
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

def predict_emotion(text):
    result = classifier(text)
    emotion = result[0]['label']
    score = result[0]['score']
    return f"Predicted Emotion: {emotion} (Confidence: {score:.2f})"

# Gradio Interface
demo = gr.Interface(fn=predict_emotion,
                    inputs=gr.Textbox(lines=4, placeholder="Enter a social media post..."),
                    outputs="text",
                    title="Emotion Decoder",
                    description="Predicts emotions like joy, sadness, anger, etc. from social media text.")

demo.launch()
