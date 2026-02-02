#!/usr/bin/env python3
# test_api.py

import requests
import json

# Test 1: Health check
print("=" * 60)
print("Testing API...")
print("=" * 60)

try:
    print("\n[1] Testing health endpoint...")
    response = requests.get("http://127.0.0.1:8000/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: Make a prediction
    print("\n[2] Testing prediction endpoint...")
    test_review = "App keeps crashing when I upload photos. Would love dark mode!"
    
    payload = {"review": test_review}
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Review: {result.get('review')}")
    print(f"Predicted Labels: {result.get('predicted_labels')}")
    print(f"Probabilities: {result.get('probabilities')}")
    
    print("\n✅ API is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure API is running: uvicorn main:app --reload --port 8000")
