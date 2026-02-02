# main.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.predict import classify_review
from app.schema import ReviewRequest, ReviewResponse

app = FastAPI(
    title="App Review Intelligence API",
    description="Multi-label NLP classification for app reviews using BERT",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url="/redoc"
)

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "App Review Intelligence API is live",
        "docs": "Visit /docs for interactive API documentation"
    }

@app.post("/predict", response_model=ReviewResponse)
def predict(request: ReviewRequest):
    """
    Classify an app review into multiple categories.
    
    **Labels:**
    - User Experience: Comments about app usability and design
    - Bug Report: Reports of technical issues
    - Ratings: Opinion or satisfaction statements
    - Feature Request: Requests for new features
    
    **Example:**
    ```
    {
        "review": "App keeps crashing on startup. Would love a dark mode feature!"
    }
    ```
    """
    try:
        result = classify_review(request.review)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)