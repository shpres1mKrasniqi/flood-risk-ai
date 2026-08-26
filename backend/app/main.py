from fastapi import FastAPI

app = FastAPI(
    title="Flood Risk Prediction API",
    description="AI-assisted flood risk prediction system for Kosovo",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {
        "message": "Flood Risk Prediction API is running."
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }