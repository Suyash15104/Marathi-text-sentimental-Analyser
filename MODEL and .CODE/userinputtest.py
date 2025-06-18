import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Load trained model and tokenizer
model_path = "./marathi-sentiment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()


label_map_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
        sentiment = label_map_reverse[pred_id]
    return sentiment

# ✅ Get user input
print("🔤 Enter Marathi sentences to analyze sentiment (type 'exit' to quit):\n")
while True:
    user_input = input("Enter sentence: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting sentiment analysis.")
        break
    sentiment = predict_sentiment(user_input)
    print(f"📊 Predicted Sentiment: {sentiment}\n")
