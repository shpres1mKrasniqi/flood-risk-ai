from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.application.flood_risk_service import FloodRiskService
from app.infrastructure.ml_risk_classifier import MLRiskClassifier
from app.infrastructure.openai_interpreter import OpenAIRiskInterpreter
from app.infrastructure.settings import get_settings
from app.presentation.routes.flood_risk import router as flood_risk_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    classifier = MLRiskClassifier(settings.flood_model_path, settings.flood_model_card_path)
    interpreter = (
        OpenAIRiskInterpreter(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
            language=settings.openai_language,
            timeout_seconds=settings.openai_timeout_seconds,
        )
        if settings.interpretation_enabled
        else None
    )
    app.state.flood_risk_service = FloodRiskService(classifier, interpreter)
    yield


app = FastAPI(
    title="Flood Risk Prediction API",
    description="AI-assisted flood risk prediction system for Kosovo",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(flood_risk_router)


@app.get("/")
async def root():
    return {"message": "Flood Risk Prediction API is running."}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
