import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ Resolve model path
model_path = Path(r"C:\Users\91860\OneDrive\Desktop\ml project\project\MODEL and .CODE\marathi-sentiment-model")

if not model_path.exists():
    raise FileNotFoundError(f"Model path not found: {model_path}")

# ✅ Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

# ✅ Label mapping
label_map_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        sentiment = label_map_reverse[pred_id]
    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
