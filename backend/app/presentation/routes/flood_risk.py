from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.application.flood_risk_service import FloodRiskService
from app.domain.flood_risk import ClassifierMetadata, InvalidFloodRiskInput
from app.presentation.dependencies import get_flood_risk_service
from app.presentation.schemas import (
    FeatureInfo,
    FloodRiskExplanationResponse,
    FloodRiskRequest,
    FloodRiskResponse,
    MetadataResponse,
    ModelInfo,
)

router = APIRouter(prefix="/api/flood-risk", tags=["flood-risk"])
ServiceDep = Annotated[FloodRiskService, Depends(get_flood_risk_service)]


def _model_info(meta: ClassifierMetadata) -> ModelInfo:
    return ModelInfo(
        name=meta.model_name, cv_macro_f1_mean=meta.cv_macro_f1_mean, cv_macro_f1_std=meta.cv_macro_f1_std
    )


@router.post("/predict", response_model=FloodRiskResponse)
def predict(request: FloodRiskRequest, service: ServiceDep) -> FloodRiskResponse:
    try:
        assessment = service.assess(request.to_domain())
    except InvalidFloodRiskInput as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return FloodRiskResponse.from_domain(assessment, _model_info(service.metadata()))


@router.post("/explain", response_model=FloodRiskExplanationResponse)
def explain(request: FloodRiskRequest, service: ServiceDep) -> FloodRiskExplanationResponse:
    """ML prediction plus a generated explanation. The class always comes from the ML model."""
    try:
        result = service.assess_and_interpret(request.to_domain())
    except InvalidFloodRiskInput as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    base = FloodRiskResponse.from_domain(result.assessment, _model_info(service.metadata()))
    return FloodRiskExplanationResponse(
        **base.model_dump(), interpretation=result.interpretation, interpretation_status=result.status.value
    )


@router.get("/metadata", response_model=MetadataResponse)
def metadata(service: ServiceDep) -> MetadataResponse:
    meta = service.metadata()
    return MetadataResponse(
        model=_model_info(meta),
        soil_types=meta.known_soil_types,
        numeric_features=[
            FeatureInfo(name=name, unit=meta.feature_units.get(name, ""), training_min=low, training_max=high)
            for name, (low, high) in meta.training_ranges.items()
        ],
        limitations=meta.limitations,
    )
