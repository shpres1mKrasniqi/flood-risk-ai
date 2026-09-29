from collections import Counter

import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ml import config
from ml.data.loader import load_dataset, split_features_target
from ml.training.cv import RANDOM_STATE, build_cv
from ml.training.evaluate import evaluate_pipeline, save_results
from ml.training.train import build_final_pipeline

OUT_DIR = config.RESULTS_DIR / "tuning"


PARAM_GRID = {
    "model__max_depth": [None, 3, 5],
    "model__min_samples_leaf": [1, 2],
}


INNER_SPLITS = 3

MACRO_F1 = make_scorer(f1_score, average="macro", zero_division=0)


def main() -> None:
    df = load_dataset()
    X, y = split_features_target(df)
    ids = df[config.ID_COLUMN]
    cv = build_cv(y)

    search = GridSearchCV(
        build_final_pipeline(),
        param_grid=PARAM_GRID,
        scoring=MACRO_F1,
        cv=StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )

    selected: list[dict] = []

    def record(model: GridSearchCV, repeat: int, fold: int) -> None:
        selected.append({"repeat": repeat, "fold": fold, **model.best_params_})

    default = evaluate_pipeline("random_forest_default", build_final_pipeline(), X, y, ids, cv)
    tuned = evaluate_pipeline("random_forest_nested_tuned", search, X, y, ids, cv, on_fold_fitted=record)
    comparison = save_results([default, tuned], OUT_DIR, cv)

    selected_df = pd.DataFrame(selected)
    selected_df.to_csv(OUT_DIR / "selected_params_per_fold.csv", index=False)
    combos = Counter(
        (str(r["model__max_depth"]), r["model__min_samples_leaf"]) for r in selected
    )
    pd.DataFrame(
        [{"max_depth": d, "min_samples_leaf": leaf, "times_selected": n} for (d, leaf), n in combos.most_common()]
    ).to_csv(OUT_DIR / "selected_params_frequency.csv", index=False)

    d, t = comparison.set_index("model").loc[["random_forest_default", "random_forest_nested_tuned"]].to_dict("records")
    gain = t["f1_macro_mean"] - d["f1_macro_mean"]
    decision = (
        "Tuning improves macro F1 by more than one standard deviation: consider the tuned settings."
        if gain > d["f1_macro_std"]
        else "Improvement is within the CV noise: keep the default settings."
    )

    print(comparison[["model", "f1_macro_mean", "f1_macro_std", "balanced_accuracy_mean",
                      "recall_low_mean", "recall_high_mean"]].to_string(index=False))
    print("\nParameters chosen by the inner search (out of 40 outer folds):")
    for (depth, leaf), n in combos.most_common():
        print(f"  max_depth={depth:<4} min_samples_leaf={leaf}: {n}")
    print(f"\nMacro F1 change: {gain:+.4f} (default std {d['f1_macro_std']})")
    print(decision)
    print(f"Results -> {OUT_DIR}")


if __name__ == "__main__":
    main()
