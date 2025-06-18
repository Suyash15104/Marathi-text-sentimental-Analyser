import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Load your testing dataset
test_df = pd.read_csv(r"C:\Users\91860\Downloads\Sentimental_dataset\Sentimental_dataset\Movie_Review_dataset_2\MahaSent_MR_Test.csv")  # replace with your file path

# ✅ Load trained model and tokenizer
model_path = "./marathi-sentiment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ Label mapping
label_map_reverse = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ✅ Predict function for a batch of texts
def predict_sentiments(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred_id = torch.argmax(logits, dim=1).item()
            results.append({
                "text": text,
                "label_id": pred_id,
                "sentiment": label_map_reverse[pred_id]
            })
    return results

# ✅ Run predictions
results = predict_sentiments(test_df["marathi_sentence"])

# ✅ Convert results to DataFrame
results_df = pd.DataFrame(results)

final_df = test_df.copy()
final_df["Predicted_Label"] = results_df["label_id"]
final_df["Predicted_Sentiment"] = results_df["sentiment"]

# ✅ Save to CSV
final_df.to_csv("marathi_test_predictions.csv", index=False)
print("✅ Predictions saved to marathi_test_predictions.csv")