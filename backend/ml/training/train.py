

import hashlib
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml import config
from ml.data.loader import load_dataset, split_features_target
from ml.preprocessing.preprocessing import build_pipeline
from ml.training.cv import RANDOM_STATE, build_cv, describe_cv
from ml.training.evaluate import evaluate_pipeline

MODEL_NAME = "random_forest"

FINAL_PARAMS: dict = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

SELECTION_REASON = (
    "Random Forest was tied with Gradient Boosting and Decision Tree for the best macro F1 "
    "(differences within one standard deviation) and had the lowest variance of the three "
    "across CV repeats. It also provides class probabilities and feature importances."
)

LIMITATIONS = [
    "Trained on 38 Kosovo municipalities only; estimates have high variance.",
    "Labels are an expert multi-criteria assessment based on the same input features, "
    "not observed flood events. The model reproduces that assessment.",
    "The Low class has 4 samples and is poorly recognised (low recall); Low predictions are unreliable.",
    "Rainfall and water-level values come from 2022 or 2024 depending on the municipality.",
    "Predicted probabilities are model scores, not real-world probabilities of flooding.",
    "Feature importances describe what the model uses, not causes of flooding.",
]


def build_final_pipeline() -> Pipeline:
    return build_pipeline(RandomForestClassifier(**FINAL_PARAMS), scale_numeric=False)


def feature_importances(pipeline: Pipeline) -> dict[str, float]:
 
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    values = pipeline.named_steps["model"].feature_importances_
    series = pd.Series(values, index=names)
    grouped = series.groupby(
        [next((c for c in config.CATEGORICAL_FEATURES if n.startswith(f"{c}_")), n) for n in names]
    ).sum()
    return {k: round(float(v), 4) for k, v in grouped.sort_values(ascending=False).items()}


def main() -> None:
    df = load_dataset()
    X, y = split_features_target(df)

    cv = build_cv(y)
    cv_summary = evaluate_pipeline(MODEL_NAME, build_final_pipeline(), X, y, df[config.ID_COLUMN], cv).summary()

  
    pipeline = build_final_pipeline().fit(X, y)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)

    reloaded = joblib.load(config.MODEL_PATH)
    assert (reloaded.predict(X) == pipeline.predict(X)).all(), "Reloaded model predicts differently."

    encoder = pipeline.named_steps["preprocessor"].named_transformers_["soil_type"]
    card = {
        "model_name": MODEL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": config.MODEL_PATH.name,
        "scikit_learn_version": sklearn.__version__,
        "dataset_file": config.DATASET_PATH.name,
        "dataset_sha256": hashlib.sha256(config.DATASET_PATH.read_bytes()).hexdigest(),
        "n_training_rows": len(df),
        "class_counts": {config.RISK_LABELS[k]: int(v) for k, v in y.value_counts().sort_index().items()},
        "target": config.TARGET,
        "classes": {str(k): v for k, v in config.RISK_LABELS.items()},
        "features": {
            "numeric": config.NUMERIC_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
            "units": config.UNITS,
            "known_soil_types": encoder.categories_[0].tolist(),
        },
        "preprocessing": "OneHotEncoder(handle_unknown='ignore') for soil_type; numeric passthrough (trees are scale-invariant).",
        "model_params": FINAL_PARAMS,
        "selection_reason": SELECTION_REASON,
        "cv": describe_cv(cv),
        "cv_estimate": cv_summary,
        "feature_importances": feature_importances(pipeline),
        "limitations": LIMITATIONS,
    }
    config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, ensure_ascii=False))

    print(f"Saved model      -> {config.MODEL_PATH}")
    print(f"Saved model card -> {config.MODEL_CARD_PATH}")
    print(f"CV macro F1: {cv_summary['f1_macro_mean']} ± {cv_summary['f1_macro_std']}")
    print("Feature importances:", card["feature_importances"])


if __name__ == "__main__":
    main()
