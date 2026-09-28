"""Evaluate a pipeline with repeated stratified CV and save the results.

Two levels of scores are produced:

* fold level   - metrics on each test fold (~9-10 rows, only 1 Low sample),
                 which makes per-fold macro F1 very noisy;
* repeat level - within one repeat every row is predicted exactly once
                 (out-of-fold), so metrics are computed on all 38 predictions.
                 This is more stable and is the level used to compare models.

Both are reported as mean +/- std so the variability is visible.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from ml import config
from ml.training.cv import describe_cv
from ml.training.metrics import LABELS, PRIMARY_METRIC, compute_metrics, pooled_confusion_matrix


@dataclass
class EvaluationResult:
    name: str
    fold_scores: pd.DataFrame
    repeat_scores: pd.DataFrame
    confusion_matrix: np.ndarray        # summed over all repeats
    predictions: pd.DataFrame           # per municipality, share of repeats per predicted class

    def summary(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {"model": self.name}
        metrics = self.repeat_scores.drop(columns="repeat")
        for column in metrics.columns:
            row[f"{column}_mean"] = round(float(metrics[column].mean()), 4)
            row[f"{column}_std"] = round(float(metrics[column].std(ddof=1)), 4)
        row[f"fold_{PRIMARY_METRIC}_mean"] = round(float(self.fold_scores[PRIMARY_METRIC].mean()), 4)
        row[f"fold_{PRIMARY_METRIC}_std"] = round(float(self.fold_scores[PRIMARY_METRIC].std(ddof=1)), 4)
        return row


def _fold_seeds(model: Pipeline, fold_index: int) -> dict[str, int]:
    """Give every fold its own fixed seed.

    Without this, a cloned estimator with random_state=42 draws the same random
    sequence in every fold, so its folds would not be independent. Seeds stay
    reproducible because they depend only on the fold index.
    """
    return {
        key: value + fold_index
        for key, value in model.get_params().items()
        if key.endswith("random_state") and isinstance(value, int)
    }


def evaluate_pipeline(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    cv: RepeatedStratifiedKFold,
) -> EvaluationResult:
    n_splits = cv.cvargs["n_splits"]
    oof = np.full((cv.n_repeats, len(y)), -1, dtype=int)
    fold_rows = []

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        repeat, fold = divmod(i, n_splits)
        model = clone(pipeline)  # fresh, unfitted copy for every fold
        model.set_params(**_fold_seeds(model, i))
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        oof[repeat, test_idx] = y_pred
        fold_rows.append({"repeat": repeat, "fold": fold, **compute_metrics(y.iloc[test_idx], y_pred)})

    assert (oof >= 0).all(), "Every row must be predicted once per repeat."

    repeat_rows = [{"repeat": r, **compute_metrics(y, oof[r])} for r in range(cv.n_repeats)]
    confusion = sum(pooled_confusion_matrix(y, oof[r]) for r in range(cv.n_repeats))

    predictions = pd.DataFrame({config.ID_COLUMN: ids.values, "true_risk": y.values})
    for label in LABELS:
        predictions[f"share_pred_{config.RISK_LABELS[label].lower()}"] = (oof == label).mean(axis=0).round(3)
    predictions["share_correct"] = (oof == y.values).mean(axis=0).round(3)

    return EvaluationResult(
        name=name,
        fold_scores=pd.DataFrame(fold_rows),
        repeat_scores=pd.DataFrame(repeat_rows),
        confusion_matrix=confusion,
        predictions=predictions,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_results(results: list[EvaluationResult], out_dir: Path, cv: RepeatedStratifiedKFold) -> pd.DataFrame:
    """Write a comparison table plus per-model details. Returns the comparison table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label_names = [config.RISK_LABELS[k] for k in LABELS]

    for result in results:
        model_dir = out_dir / result.name
        model_dir.mkdir(exist_ok=True)
        result.fold_scores.round(4).to_csv(model_dir / "fold_scores.csv", index=False)
        result.repeat_scores.round(4).to_csv(model_dir / "repeat_scores.csv", index=False)
        pd.DataFrame(
            result.confusion_matrix,
            index=[f"true_{n}" for n in label_names],
            columns=[f"pred_{n}" for n in label_names],
        ).to_csv(model_dir / "confusion_matrix.csv")
        result.predictions.to_csv(model_dir / "predictions.csv", index=False)

    comparison = pd.DataFrame([r.summary() for r in results])
    comparison = comparison.sort_values(f"{PRIMARY_METRIC}_mean", ascending=False)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    run_info = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_file": config.DATASET_PATH.name,
        "dataset_sha256": _sha256(config.DATASET_PATH),
        "scikit_learn_version": sklearn.__version__,
        "cv": describe_cv(cv),
        "primary_metric": PRIMARY_METRIC,
        "confusion_matrix_note": f"Counts summed over {cv.n_repeats} repeats; each row is predicted once per repeat.",
        "models": [r.name for r in results],
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    return comparison
