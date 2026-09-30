from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.application.flood_risk_service import FloodRiskService
from app.infrastructure.ml_risk_classifier import MLRiskClassifier
from app.infrastructure.settings import get_settings
from app.presentation.routes.flood_risk import router as flood_risk_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    classifier = MLRiskClassifier(settings.flood_model_path, settings.flood_model_card_path)
    app.state.flood_risk_service = FloodRiskService(classifier)
    yield


app = FastAPI(
    title="Flood Risk Prediction API",
    description="AI-assisted flood risk prediction system for Kosovo",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(flood_risk_router)


@app.get("/")
async def root():
    return {"message": "Flood Risk Prediction API is running."}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
