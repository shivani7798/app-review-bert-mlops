# app/predict.py

from .model import load_model, predict_labels
from .preprocess import clean_text

# Load model ONCE at startup
print("Loading model...")
_tokenizer, _model = load_model()
print("✓ Model loaded successfully")

def classify_review(review: str):
    """
    Classify an app review into multiple labels.
    
    Args:
        review: The review text to classify
        
    Returns:
        dict with review, predicted_labels, and probabilities
    """
    # Preprocess and predict
    clean_review = clean_text(review)
    labels, probs = predict_labels(clean_review, _tokenizer, _model)
    
    return {
        "review": clean_review,
        "predicted_labels": labels,
        "probabilities": probs,
        "label_names": ["feature request", "bug report", "rating", "user experience"]
    }
