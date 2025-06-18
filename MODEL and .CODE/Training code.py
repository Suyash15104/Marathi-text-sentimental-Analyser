
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ✅ Load and Prepare Data
df = pd.read_csv(r"C:\Users\91860\Downloads\Sentimental_dataset\Sentimental_dataset\Movie_Review_dataset_2\MahaSent_MR_Train.csv")  # Replace with your file path
df = df[['marathi_sentence', 'label']]
df = df.rename(columns={'marathi_sentence': 'Text', 'label': 'Label'})

# Map labels from -1, 0, 1 → 0, 1, 2
label_map = {-1: 0, 0: 1, 1: 2}
df['Label'] = df['Label'].map(label_map)

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Rename 'Label' to 'labels' for Hugging Face
train_df = train_df.rename(columns={'Label': 'labels'})
test_df = test_df.rename(columns={'Label': 'labels'})

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ✅ Tokenization
model_name = "google/muril-base-cased"  # or "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["Text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ✅ Load Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# ✅ Define Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}

# ✅ Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=10,
    save_total_limit=1,
)

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ✅ Train the Model
trainer.train()

# ✅ Evaluate
trainer.evaluate()

# ✅ Save the Model and Tokenizer
model.save_pretrained("./marathi-sentiment-model")
tokenizer.save_pretrained("./marathi-sentiment-model")
print("✅ Model and tokenizer saved to ./marathi-sentiment-model")