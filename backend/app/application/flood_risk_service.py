from app.domain.flood_risk import (
    ClassifierMetadata,
    FloodRiskAssessment,
    FloodRiskInput,
    InvalidFloodRiskInput,
    RiskClassifier,
)


class FloodRiskService:
    def __init__(self, classifier: RiskClassifier) -> None:
        self._classifier = classifier

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
