import json
from pathlib import Path

from app.domain.flood_risk import (
    ClassifierMetadata,
    FloodRiskInput,
    RiskClassification,
    RiskClassifier,
    RiskLevel,
)
from ml.inference.predictor import FloodRiskPredictor


class MLRiskClassifier(RiskClassifier):
    def __init__(self, model_path: Path, model_card_path: Path) -> None:
        self._predictor = FloodRiskPredictor(model_path)
        self._metadata = self._load_metadata(Path(model_card_path))

    def classify(self, data: FloodRiskInput) -> RiskClassification:
        prediction = self._predictor.predict(data.features())
        label_to_level = {level.label: level for level in RiskLevel}
        return RiskClassification(
            risk_level=RiskLevel(prediction.risk_code),
            class_scores={label_to_level[k]: v for k, v in prediction.class_scores.items()},
        )

    def metadata(self) -> ClassifierMetadata:
        return self._metadata

    def _load_metadata(self, card_path: Path) -> ClassifierMetadata:
        if not card_path.exists():
            raise FileNotFoundError(f"Model card not found: {card_path}")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        features = card["features"]
        return ClassifierMetadata(
            model_name=card["model_name"],
            known_soil_types=self._predictor.known_soil_types,
            feature_units=features["units"],
            training_ranges={f: (r["min"], r["max"]) for f, r in features.get("training_ranges", {}).items()},
            cv_macro_f1_mean=card["cv_estimate"]["f1_macro_mean"],
            cv_macro_f1_std=card["cv_estimate"]["f1_macro_std"],
            feature_importances=card["feature_importances"],
            limitations=card["limitations"],
        )
