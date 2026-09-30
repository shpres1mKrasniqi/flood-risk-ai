"""Application layer: orchestrates the flood-risk use cases.

Depends only on the domain. It does not know whether the classifier is a
Random Forest or a test fake, nor whether the interpreter is OpenAI.
"""

import logging
from dataclasses import dataclass
from enum import Enum

from app.domain.flood_risk import (
    ClassifierMetadata,
    FloodRiskAssessment,
    FloodRiskInput,
    InterpretationUnavailable,
    InvalidFloodRiskInput,
    RiskClassifier,
    RiskInterpreter,
)

logger = logging.getLogger(__name__)


class InterpretationStatus(str, Enum):
    OK = "ok"
    DISABLED = "disabled"          # no interpreter configured (e.g. no API key)
    UNAVAILABLE = "unavailable"    # interpreter failed (network, quota, timeout)
    REJECTED = "rejected"          # text did not confirm the ML class, so it was discarded


@dataclass(frozen=True)
class InterpretedAssessment:
    assessment: FloodRiskAssessment
    interpretation: str | None
    status: InterpretationStatus


def interpretation_header(assessment: FloodRiskAssessment) -> str:
    """First line every interpretation must start with; used to verify it kept the ML class."""
    return f"Risk level: {assessment.classification.risk_level.label}"


class FloodRiskService:
    def __init__(self, classifier: RiskClassifier, interpreter: RiskInterpreter | None = None) -> None:
        self._classifier = classifier
        self._interpreter = interpreter

    def assess(self, data: FloodRiskInput) -> FloodRiskAssessment:
        meta = self._classifier.metadata()
        if data.soil_type not in meta.known_soil_types:
            raise InvalidFloodRiskInput(
                f"Unknown soil_type '{data.soil_type}'. Allowed: {', '.join(meta.known_soil_types)}."
            )
        classification = self._classifier.classify(data)
        return FloodRiskAssessment(
            input=data,
            classification=classification,
            warnings=self._out_of_range_warnings(data, meta),
        )

    def assess_and_interpret(self, data: FloodRiskInput) -> InterpretedAssessment:
        """ML prediction first; the interpretation is added afterwards and can never change it.

        If interpretation fails, the prediction is still returned.
        """
        assessment = self.assess(data)
        if self._interpreter is None:
            return InterpretedAssessment(assessment, None, InterpretationStatus.DISABLED)

        try:
            text = self._interpreter.interpret(assessment, self._classifier.metadata()).strip()
        except InterpretationUnavailable as exc:
            logger.warning("Interpretation unavailable: %s", exc)
            return InterpretedAssessment(assessment, None, InterpretationStatus.UNAVAILABLE)

        first_line = text.splitlines()[0].strip() if text else ""
        if first_line.lower() != interpretation_header(assessment).lower():
            logger.warning("Interpretation rejected: first line %r does not confirm the ML class.", first_line)
            return InterpretedAssessment(assessment, None, InterpretationStatus.REJECTED)

        return InterpretedAssessment(assessment, text, InterpretationStatus.OK)

    def metadata(self) -> ClassifierMetadata:
        return self._classifier.metadata()

    @staticmethod
    def _out_of_range_warnings(data: FloodRiskInput, meta: ClassifierMetadata) -> list[str]:
        """Flag values outside the training data: the model is extrapolating there."""
        warnings = []
        for feature, (low, high) in meta.training_ranges.items():
            value = getattr(data, feature)
            if not low <= value <= high:
                unit = meta.feature_units.get(feature, "")
                warnings.append(
                    f"{feature}={value:g} {unit} is outside the training range "
                    f"[{low:g}, {high:g}] {unit}; the prediction is less reliable."
                )
        return warnings
