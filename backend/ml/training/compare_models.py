"""Phase 5 - compare candidate models with default/conservative settings (no tuning).

Run from the backend/ directory:
    uv run --group ml python -m ml.training.compare_models

Every model is evaluated with exactly the same CV splits, metrics and
preprocessing as the baseline. Two feature sets are run:
* full_features                - all 8 predictive features;
* without_distance_from_river  - ablation, to measure that feature's contribution.
"""

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml import config
from ml.data.loader import load_dataset, split_features_target
from ml.preprocessing.preprocessing import build_pipeline
from ml.training.cv import RANDOM_STATE, build_cv, describe_cv
from ml.training.evaluate import EvaluationResult, evaluate_pipeline, save_results
from ml.training.metrics import LABELS, PRIMARY_METRIC

OUT_DIR = config.RESULTS_DIR / "comparison"
PLOT_DIR = config.PLOTS_DIR / "comparison"


@dataclass(frozen=True)
class Candidate:
    model: BaseEstimator
    scale_numeric: bool
    rationale: str


def candidates() -> dict[str, Candidate]:
    """Default or deliberately conservative settings; nothing here is tuned."""
    return {
        "dummy_most_frequent": Candidate(
            DummyClassifier(strategy="most_frequent"), True, "Reference: always predicts Medium."
        ),
        "logistic_regression": Candidate(
            LogisticRegression(class_weight="balanced", max_iter=5000),
            True,
            "Linear, interpretable; balanced weights counter the Medium majority.",
        ),
        "decision_tree": Candidate(
            DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE),
            False,
            "Interpretable rules; depth 3 limits memorising 38 rows.",
        ),
        "random_forest": Candidate(
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
            False,
            "Averaged trees reduce the variance of a single tree.",
        ),
        "svm_rbf": Candidate(
            SVC(kernel="rbf", class_weight="balanced"),
            True,
            "Non-linear boundary; needs scaled features.",
        ),
        "knn": Candidate(
            KNeighborsClassifier(n_neighbors=3),
            True,
            "k=3 because Low has only 3 training samples per fold; no class weighting available.",
        ),
        "gradient_boosting": Candidate(
            HistGradientBoostingClassifier(
                max_depth=2, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE
            ),
            False,
            "Shallow boosting; min_samples_leaf lowered from 20 (default) because of 38 rows.",
        ),
    }


def run_feature_set(
    label: str, exclude: tuple[str, ...], X: pd.DataFrame, y: pd.Series, ids: pd.Series
) -> tuple[pd.DataFrame, list[EvaluationResult]]:
    cv = build_cv(y)
    results = [
        evaluate_pipeline(name, build_pipeline(c.model, c.scale_numeric, exclude), X, y, ids, cv)
        for name, c in candidates().items()
    ]
    comparison = save_results(results, OUT_DIR / label, cv)
    return comparison, results


def misclassification_overview(results: list[EvaluationResult]) -> pd.DataFrame:
    """Share of repeats each municipality was predicted correctly, per model.

    Rows that almost every model gets wrong are candidates for label review.
    """
    base = results[0].predictions[[config.ID_COLUMN, "true_risk"]].copy()
    for r in results:
        if r.name.startswith("dummy"):
            continue
        base[r.name] = r.predictions["share_correct"].values
    model_columns = [c for c in base.columns if c not in (config.ID_COLUMN, "true_risk")]
    base["mean_share_correct"] = base[model_columns].mean(axis=1).round(3)
    base["true_label"] = base["true_risk"].map(config.RISK_LABELS)
    return base.sort_values("mean_share_correct")


def plot_repeat_scores(results: list[EvaluationResult], path) -> None:
    data = [r.repeat_scores[PRIMARY_METRIC] for r in results]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(data, tick_labels=[r.name for r in results])
    ax.set_ylabel("Macro F1 per repeat (out-of-fold)")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison - repeated stratified CV")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(results: list[EvaluationResult], path) -> None:
    names = [config.RISK_LABELS[k] for k in LABELS]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, r in zip(axes.flat, results):
        cm = r.confusion_matrix
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), names)
        ax.set_title(r.name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    for ax in list(axes.flat)[len(results):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (counts summed over repeats; colour = row share)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X, y = split_features_target(df)
    ids = df[config.ID_COLUMN]

    full, full_results = run_feature_set("full_features", (), X, y, ids)
    ablation, _ = run_feature_set("without_distance_from_river", ("distance_from_river",), X, y, ids)

    misclassification_overview(full_results).to_csv(OUT_DIR / "misclassification_overview.csv", index=False)
    plot_repeat_scores(full_results, PLOT_DIR / "macro_f1_by_model.png")
    plot_confusion_matrices(full_results, PLOT_DIR / "confusion_matrices.png")

    key = [f"{PRIMARY_METRIC}_mean", f"{PRIMARY_METRIC}_std"]
    delta = full[["model", *key]].merge(
        ablation[["model", *key]], on="model", suffixes=("_full", "_without_distance")
    )
    delta["delta_f1_macro"] = (
        delta[f"{PRIMARY_METRIC}_mean_without_distance"] - delta[f"{PRIMARY_METRIC}_mean_full"]
    ).round(4)
    delta.to_csv(OUT_DIR / "ablation_distance_from_river.csv", index=False)

    pd.DataFrame(
        [{"model": n, "scale_numeric": c.scale_numeric, "params": str(c.model.get_params()), "rationale": c.rationale}
         for n, c in candidates().items()]
    ).to_csv(OUT_DIR / "model_settings.csv", index=False)

    columns = [
        "model", "f1_macro_mean", "f1_macro_std", "balanced_accuracy_mean",
        "recall_low_mean", "recall_high_mean", "accuracy_mean",
    ]
    pd.set_option("display.width", 200)
    print(f"CV: {describe_cv(build_cv(y))}\n")
    print("Full features:")
    print(full[columns].to_string(index=False))
    print("\nAblation (without distance_from_river), change in macro F1:")
    print(delta[["model", "f1_macro_mean_full", "f1_macro_mean_without_distance", "delta_f1_macro"]].to_string(index=False))
    print(f"\nResults -> {OUT_DIR}\nPlots   -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
