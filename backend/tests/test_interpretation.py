
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.application.flood_risk_service import FloodRiskService, InterpretationStatus
from app.domain.flood_risk import (
    FloodRiskAssessment,
    FloodRiskInput,
    InterpretationUnavailable,
    RiskInterpreter,
    RiskLevel,
)
from app.infrastructure.openai_interpreter import OpenAIRiskInterpreter, build_context
from app.main import app
from tests.test_flood_risk_service import VALID, FakeClassifier  # fake classifier always predicts High


class TextInterpreter(RiskInterpreter):
    def __init__(self, text: str) -> None:
        self.text = text

    def interpret(self, assessment, metadata) -> str:
        return self.text


class FailingInterpreter(RiskInterpreter):
    def interpret(self, assessment, metadata) -> str:
        raise InterpretationUnavailable("timeout")


def _service(interpreter):
    return FloodRiskService(FakeClassifier(), interpreter)


# --- application rules -------------------------------------------------------

def test_interpretation_is_returned_when_it_confirms_the_ml_class():
    result = _service(TextInterpreter("Risk level: High\nExplanation...")).assess_and_interpret(FloodRiskInput(**VALID))
    assert result.status is InterpretationStatus.OK
    assert result.interpretation.startswith("Risk level: High")


def test_interpretation_that_states_another_class_is_rejected():
    result = _service(TextInterpreter("Risk level: Low\nActually low.")).assess_and_interpret(FloodRiskInput(**VALID))
    assert result.status is InterpretationStatus.REJECTED
    assert result.interpretation is None
    assert result.assessment.classification.risk_level is RiskLevel.HIGH  # ML class is untouched


def test_prediction_survives_interpreter_failure():
    result = _service(FailingInterpreter()).assess_and_interpret(FloodRiskInput(**VALID))
    assert result.status is InterpretationStatus.UNAVAILABLE
    assert result.assessment.classification.risk_level is RiskLevel.HIGH


def test_no_interpreter_means_disabled():
    result = _service(None).assess_and_interpret(FloodRiskInput(**VALID))
    assert result.status is InterpretationStatus.DISABLED


# --- what is sent to OpenAI --------------------------------------------------

def _assessment(**override) -> FloodRiskAssessment:
    service = _service(None)
    return service.assess(FloodRiskInput(**VALID | override, municipality="Peja"))


def test_context_contains_only_this_prediction():
    context = build_context(_assessment(), FakeClassifier().metadata())
    assert set(context) == {"municipality", "predicted_risk_level", "class_scores", "input_values", "warnings", "model"}
    assert context["predicted_risk_level"] == "High"
    assert len(context["input_values"]) == 8


# --- OpenAI adapter with a mocked client -------------------------------------

def _interpreter_with(create):
    interpreter = OpenAIRiskInterpreter(api_key="test-key", model="test-model")
    interpreter._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return interpreter


def test_adapter_sends_expected_request():
    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(output_text="Risk level: High\nText")

    text = _interpreter_with(create).interpret(_assessment(), FakeClassifier().metadata())
    assert text.startswith("Risk level: High")
    assert calls["model"] == "test-model"
    assert calls["store"] is False
    assert '"Risk level: High"' in calls["instructions"]


def test_adapter_maps_openai_errors():
    def create(**kwargs):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))

    with pytest.raises(InterpretationUnavailable):
        _interpreter_with(create).interpret(_assessment(), FakeClassifier().metadata())


# --- endpoint ------------------------------------------------------------------

def test_explain_endpoint_returns_prediction_and_interpretation():
    decan = dict(elevation=678, distance_from_river=0.79, rainfall=800, soil_type="Aluviale",
                 max_water_level=97, min_water_level=27, min_slope=3, max_slope=70)
    with TestClient(app) as client:
        real_service = app.state.flood_risk_service
        # Keep the real classifier, replace only the interpreter with a fake.
        app.state.flood_risk_service = FloodRiskService(real_service._classifier, TextInterpreter("Risk level: Medium\nOK"))
        try:
            body = client.post("/api/flood-risk/explain", json=decan).json()
        finally:
            app.state.flood_risk_service = real_service
    assert body["risk"] == "Medium"
    assert body["interpretation_status"] == "ok"
    assert body["interpretation"].startswith("Risk level: Medium")
