from fastapi import Request

from app.application.flood_risk_service import FloodRiskService


def get_flood_risk_service(request: Request) -> FloodRiskService:
    return request.app.state.flood_risk_service
