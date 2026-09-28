"""Phase 4 - baseline classifiers.

Run from the backend/ directory:
    uv run --group ml python -m ml.training.baseline

Two trivial classifiers define the minimum a real model must beat:
* most_frequent - always predicts the majority class (Medium);
* stratified    - guesses randomly in the training-set class proportions.
"""

from sklearn.dummy import DummyClassifier

from ml import config
from ml.data.loader import load_dataset, split_features_target
from ml.preprocessing.preprocessing import build_pipeline
from ml.training.cv import RANDOM_STATE, build_cv, describe_cv
from ml.training.evaluate import evaluate_pipeline, save_results

OUT_DIR = config.RESULTS_DIR / "baseline"

BASELINES = {
    "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
    "dummy_stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
}


def main() -> None:
    df = load_dataset()
    X, y = split_features_target(df)
    cv = build_cv(y)

    results = [
        evaluate_pipeline(name, build_pipeline(model), X, y, df[config.ID_COLUMN], cv)
        for name, model in BASELINES.items()
    ]
    comparison = save_results(results, OUT_DIR, cv)

    print(f"CV: {describe_cv(cv)}")
    columns = ["model", "f1_macro_mean", "f1_macro_std", "balanced_accuracy_mean", "accuracy_mean", "f1_weighted_mean"]
    print(comparison[columns].to_string(index=False))
    print(f"Results -> {OUT_DIR}")


if __name__ == "__main__":
    main()
