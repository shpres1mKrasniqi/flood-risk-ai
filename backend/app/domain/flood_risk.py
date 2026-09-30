from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum


class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

    @property
    def label(self) -> str:
        return self.name.capitalize()


class InvalidFloodRiskInput(ValueError):
    """Input that violates a domain rule (e.g. min > max, unknown soil type)."""


@dataclass(frozen=True)
class FloodRiskInput:
    elevation: float
    distance_from_river: float
    rainfall: float
    soil_type: str
    max_water_level: float
    min_water_level: float
    min_slope: float
    max_slope: float
    municipality: str | None = None

    def __post_init__(self) -> None:
        if self.min_water_level > self.max_water_level:
            raise InvalidFloodRiskInput("min_water_level cannot be greater than max_water_level.")
        if self.min_slope > self.max_slope:
            raise InvalidFloodRiskInput("min_slope cannot be greater than max_slope.")

    def features(self) -> dict[str, float | str]:
        """The eight predictive features, without the municipality name."""
        return {
            "elevation": self.elevation,
            "distance_from_river": self.distance_from_river,
            "rainfall": self.rainfall,
            "soil_type": self.soil_type,
            "max_water_level": self.max_water_level,
            "min_water_level": self.min_water_level,
            "min_slope": self.min_slope,
            "max_slope": self.max_slope,
        }


@dataclass(frozen=True)
class RiskClassification:
    """What a classifier returns: the class and its per-class model scores."""

    risk_level: RiskLevel

    class_scores: dict[RiskLevel, float]


@dataclass(frozen=True)
class ClassifierMetadata:
    model_name: str
    known_soil_types: list[str]
    feature_units: dict[str, str]
    training_ranges: dict[str, tuple[float, float]]
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    feature_importances: dict[str, float]
    limitations: list[str]


@dataclass(frozen=True)
class FloodRiskAssessment:
    input: FloodRiskInput
    classification: RiskClassification
    warnings: list[str] = field(default_factory=list)


class RiskClassifier(ABC):

    @abstractmethod
    def classify(self, data: FloodRiskInput) -> RiskClassification: ...

    @abstractmethod
    def metadata(self) -> ClassifierMetadata: ...


class InterpretationUnavailable(RuntimeError):
    """The interpreter could not produce an explanation (network, quota, timeout, ...)."""


class RiskInterpreter(ABC):
    """Port: turns an ML assessment into a human-readable explanation.

    An interpreter explains a classification; it never decides or changes it.
    """

    @abstractmethod
    def interpret(self, assessment: "FloodRiskAssessment", metadata: ClassifierMetadata) -> str: ...
