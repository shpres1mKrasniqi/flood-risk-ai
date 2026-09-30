import pytest

from app.application.flood_risk_service import FloodRiskService
from app.domain.flood_risk import (
    ClassifierMetadata,
    FloodRiskInput,
    InvalidFloodRiskInput,
    RiskClassification,
    RiskClassifier,
    RiskLevel,
)

VALID = dict(
    elevation=500, distance_from_river=1.0, rainfall=700, soil_type="Smonice",
    max_water_level=150, min_water_level=30, min_slope=3, max_slope=30,
)


class FakeClassifier(RiskClassifier):
    def __init__(self) -> None:
        self.received: FloodRiskInput | None = None

    def classify(self, data: FloodRiskInput) -> RiskClassification:
        self.received = data
        return RiskClassification(RiskLevel.HIGH, {RiskLevel.LOW: 0.1, RiskLevel.MEDIUM: 0.2, RiskLevel.HIGH: 0.7})

    def metadata(self) -> ClassifierMetadata:
        return ClassifierMetadata(
            model_name="fake", known_soil_types=["Smonice"], feature_units={"rainfall": "mm"},
            training_ranges={"rainfall": (300.0, 900.0)}, cv_macro_f1_mean=0.5, cv_macro_f1_std=0.1,
            feature_importances={}, limitations=[],
        )


def test_service_returns_classifier_result():
    service = FloodRiskService(FakeClassifier())
    assessment = service.assess(FloodRiskInput(**VALID))
    assert assessment.classification.risk_level is RiskLevel.HIGH
    assert assessment.warnings == []


def test_unknown_soil_type_is_rejected_before_classification():
    classifier = FakeClassifier()
    with pytest.raises(InvalidFloodRiskInput, match="Unknown soil_type"):
        FloodRiskService(classifier).assess(FloodRiskInput(**VALID | {"soil_type": "Sand"}))
    assert classifier.received is None


def test_value_outside_training_range_produces_warning():
    assessment = FloodRiskService(FakeClassifier()).assess(FloodRiskInput(**VALID | {"rainfall": 1500}))
    assert len(assessment.warnings) == 1
    assert "rainfall" in assessment.warnings[0]


@pytest.mark.parametrize(
    "override", [{"min_water_level": 200, "max_water_level": 100}, {"min_slope": 40, "max_slope": 10}]
)
def test_min_greater_than_max_is_invalid(override):
    with pytest.raises(InvalidFloodRiskInput):
        FloodRiskInput(**VALID | override)


def test_municipality_is_not_a_model_feature():
    features = FloodRiskInput(**VALID, municipality="Peja").features()
    assert "municipality" not in features
    assert len(features) == 8


def test_risk_level_labels():
    assert [level.label for level in RiskLevel] == ["Low", "Medium", "High"]
