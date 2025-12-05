import os

os.system("pip install numpy==1.26.4")
os.system("pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html")
os.system("pip install transformers==4.37.2")

import torch
import gradio as gr
from transformers import DebertaV2TokenizerFast, AutoModelForSequenceClassification

model_name = "Deepak8409/deberta-emotion-classifier"
labels = ["anger", "fear", "joy", "sadness", "surprise"]

tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        logits = model(**inputs).logits

        probs = torch.sigmoid(logits).detach().numpy()[0]

        scores = {label: float(p) for label, p in zip(labels, probs)}
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return scores

    except Exception as e:
        return {"error": str(e)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter text..."),
    outputs="label",
    title="Emotion Classifier",
    description="Multi-label emotion prediction using DeBERTa v3"
)

demo.launch()


