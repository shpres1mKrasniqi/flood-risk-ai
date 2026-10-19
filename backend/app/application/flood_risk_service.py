import logging
from dataclasses import dataclass
from enum import Enum

from app.domain.flood_risk import (
    ClassifierMetadata,
    FloodRiskAssessment,
    FloodRiskInput,
    InterpretationUnavailable,
    InvalidFloodRiskInput,
    RangeViolation,
    RiskClassifier,
    RiskInterpreter,
)

logger = logging.getLogger(__name__)


class InterpretationStatus(str, Enum):
    OK = "ok"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


@dataclass(frozen=True)
class InterpretedAssessment:
    assessment: FloodRiskAssessment
    interpretation: str | None
    status: InterpretationStatus


def interpretation_header(assessment: FloodRiskAssessment) -> str:
    """First line every interpretation must start with; used to verify it kept the ML class."""
    return f"Risk level: {assessment.classification.risk_level.label}"


def _normalise_header(line: str) -> str:
    """Ignore markdown and punctuation around the header, e.g. '**Risk level: High.**'."""
    return " ".join(line.strip("*_#`\"'. \t").lower().split())


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
            range_violations=self._range_violations(data, meta),
        )

    def assess_and_interpret(self, data: FloodRiskInput, language: str | None = None) -> InterpretedAssessment:
        """ML prediction first; the interpretation is added afterwards and can never change it.

        If interpretation fails, the prediction is still returned.
        """
        assessment = self.assess(data)
        if self._interpreter is None:
            return InterpretedAssessment(assessment, None, InterpretationStatus.DISABLED)

        try:
            text = self._interpreter.interpret(assessment, self._classifier.metadata(), language).strip()
        except InterpretationUnavailable as exc:
            logger.warning("Interpretation unavailable: %s", exc)
            return InterpretedAssessment(assessment, None, InterpretationStatus.UNAVAILABLE)

        lines = text.splitlines()
        first_line = lines[0].strip() if lines else ""
        header = interpretation_header(assessment)
        if _normalise_header(first_line) != header.lower():
            logger.warning("Interpretation rejected: first line %r does not confirm the ML class.", first_line)
            return InterpretedAssessment(assessment, None, InterpretationStatus.REJECTED)

        # Replace the model's header with the canonical one (drops markdown/punctuation variations).
        body = "\n".join(lines[1:]).strip()
        return InterpretedAssessment(assessment, f"{header}\n\n{body}".strip(), InterpretationStatus.OK)

    def metadata(self) -> ClassifierMetadata:
        return self._classifier.metadata()

    @staticmethod
    def _range_violations(data: FloodRiskInput, meta: ClassifierMetadata) -> list[RangeViolation]:
        """Flag values outside the training data: the model is extrapolating there."""
        violations = []
        for feature, (low, high) in meta.training_ranges.items():
            value = getattr(data, feature)
            if not low <= value <= high:
                violations.append(RangeViolation(feature, value, low, high, meta.feature_units.get(feature, "")))
        return violations
