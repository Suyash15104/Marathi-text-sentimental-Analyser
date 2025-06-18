from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model_path = "./marathi-sentiment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_map_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        sentiment = label_map_reverse[pred_id]
    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
