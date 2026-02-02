#!/usr/bin/env python3
# train/train.py

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split

# Labels (MUST match your CSV column names exactly)
LABELS = ['feature request', 'bug report', 'rating', 'user experience']
DATA_PATH = "data/sample_reviews.csv"
MODEL_OUTPUT_DIR = "saved_model"
RESULTS_DIR = "./results"

print("=" * 60)
print("🚀 BERT Multi-Label Classification Training")
print("=" * 60)

# Step 1: Load data
print("\n[1/5] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df)} reviews")
print(f"✓ Columns: {df.columns.tolist()}")

# Step 2: Prepare data
print("\n[2/5] Preparing data...")
train_texts = df['review'].tolist()
train_labels = df[LABELS].values.tolist()

# Split into train/validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=42
)

print(f"✓ Training set: {len(train_texts)} reviews")
print(f"✓ Validation set: {len(val_texts)} reviews")

# Step 3: Tokenize
print("\n[3/5] Tokenizing texts...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

print(f"✓ Tokenization complete")

# Step 4: Create Dataset class
class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = MultiLabelDataset(train_encodings, train_labels)
val_dataset = MultiLabelDataset(val_encodings, val_labels)

print(f"✓ Dataset objects created")

# Step 5: Train model
print("\n[4/5] Training BERT model...")
print(f"✓ Using GPU: {torch.cuda.is_available()}")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
)
training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Step 6: Save model
print("\n[5/5] Saving trained model...")
Path(MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

print(f"✓ Model saved to {MODEL_OUTPUT_DIR}")
print("\n" + "=" * 60)
print("✅ Training complete!")
print("=" * 60)
print(f"\nYour API will now use the trained model.")
print(f"Restart the API with: uvicorn main:app --reload")
