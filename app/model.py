# app/model.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification

LABELS = ['User Experience', 'Bug Report', 'Ratings', 'Feature Request']

# Global variables to cache the model (load only once)
_tokenizer = None
_model = None

def load_model(model_path="bert-base-uncased", local_path="saved_model"):
    """
    Load the tokenizer and model.
    If local_path exists, load from there.
    Otherwise, use the pre-trained model_path from HuggingFace.
    """
    global _tokenizer, _model
    
    # Return cached model if already loaded
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    
    try:
        # Try to load from local saved_model folder first
        _tokenizer = BertTokenizer.from_pretrained(local_path)
        _model = BertForSequenceClassification.from_pretrained(local_path)
        print(f"✓ Loaded model from {local_path}")
    except:
        # Fall back to pre-trained model from HuggingFace
        print(f"⚠ Local model not found, using pre-trained {model_path}")
        _tokenizer = BertTokenizer.from_pretrained(model_path)
        _model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(LABELS),
            problem_type="multi_label_classification"
        )
    
    _model.eval()
    return _tokenizer, _model

def predict_labels(text, tokenizer, model, threshold=0.5):
    """Predict labels for input text using the model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    
    # Handle single label case (convert to list)
    if not isinstance(probs, list):
        probs = [probs]
    
    predictions = [LABELS[i] for i, p in enumerate(probs) if p > threshold]
    return predictions, probs
