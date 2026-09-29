from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ml import config


@dataclass(frozen=True)
class RiskPrediction:
    risk_code: int
    risk_label: str

    class_scores: dict[str, float]


class FloodRiskPredictor:
    def __init__(self, model_path: Path | str = config.MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Train it with: python -m ml.training.train"
            )
        self._pipeline = joblib.load(model_path)
        encoder = self._pipeline.named_steps["preprocessor"].named_transformers_["soil_type"]
        self._known_soil_types: frozenset[str] = frozenset(encoder.categories_[0].tolist())

    @property
    def known_soil_types(self) -> list[str]:
        return sorted(self._known_soil_types)

    def predict(self, features: Mapping[str, float | str]) -> RiskPrediction:
        missing = [f for f in config.FEATURES if f not in features]
        if missing:
            raise ValueError(f"Missing feature(s): {missing}")

        
        soil_type = str(features["soil_type"]).strip()
        if soil_type not in self._known_soil_types:
            raise ValueError(f"Unknown soil_type {soil_type!r}. Known: {self.known_soil_types}")

        row = {f: features[f] for f in config.FEATURES} | {"soil_type": soil_type}
        X = pd.DataFrame([row], columns=config.FEATURES)

        code = int(self._pipeline.predict(X)[0])
        scores = self._pipeline.predict_proba(X)[0]
        classes = self._pipeline.classes_
        return RiskPrediction(
            risk_code=code,
            risk_label=config.RISK_LABELS[code],
            class_scores={config.RISK_LABELS[int(c)]: round(float(s), 3) for c, s in zip(classes, scores)},
        )
