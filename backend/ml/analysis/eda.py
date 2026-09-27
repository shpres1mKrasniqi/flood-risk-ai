import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd

from ml import config
from ml.data.loader import load_dataset

RESULTS = config.RESULTS_DIR / "eda"
PLOTS = config.PLOTS_DIR / "eda"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[config.TARGET].value_counts().reindex(config.RISK_LABELS.keys(), fill_value=0)
    return pd.DataFrame(
        {
            "risk": counts.index,
            "label": [config.RISK_LABELS[c] for c in counts.index],
            "count": counts.values,
            "share": (counts.values / len(df)).round(4),
        }
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df[config.NUMERIC_FEATURES].describe().T
    summary["skew"] = df[config.NUMERIC_FEATURES].skew()
    summary.insert(0, "unit", [config.UNITS[c] for c in summary.index])
    return summary.round(3).rename_axis("feature").reset_index()


def numeric_by_class(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(config.TARGET)[config.NUMERIC_FEATURES].agg(["median", "min", "max"])
    grouped.index = [f"{i} ({config.RISK_LABELS[i]})" for i in grouped.index]
    return grouped.T.round(3).rename_axis(["feature", "statistic"]).reset_index()


def iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag values outside 1.5 x IQR. Descriptive only: with 38 rows this is a screening aid."""
    rows = []
    for feature in config.NUMERIC_FEATURES:
        q1, q3 = df[feature].quantile([0.25, 0.75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        flagged = df[(df[feature] < lower) | (df[feature] > upper)]
        for _, r in flagged.iterrows():
            rows.append(
                {
                    "municipality": r[config.ID_COLUMN],
                    "feature": feature,
                    "value": r[feature],
                    "lower_fence": round(lower, 3),
                    "upper_fence": round(upper, 3),
                    "risk": r[config.TARGET],
                }
            )
    return pd.DataFrame(rows, columns=["municipality", "feature", "value", "lower_fence", "upper_fence", "risk"])


def cv_feasibility(df: pd.DataFrame) -> dict:
    counts = df[config.TARGET].value_counts()
    smallest = int(counts.min())
    return {
        "n_rows": len(df),
        "class_counts": {config.RISK_LABELS[k]: int(v) for k, v in counts.sort_index().items()},
        "smallest_class": config.RISK_LABELS[int(counts.idxmin())],
        "smallest_class_count": smallest,
        "max_stratified_folds": smallest,
        "note": (
            "StratifiedKFold needs at least n_splits samples in every class, so the number of "
            "folds cannot exceed the smallest class count."
        ),
    }


def save_plots(df: pd.DataFrame, dist: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(dist["label"], dist["count"], color=["#4c9f70", "#e0a030", "#c0392b"])
    for i, count in enumerate(dist["count"]):
        ax.text(i, count + 0.3, str(count), ha="center")
    ax.set_ylabel("Number of municipalities")
    ax.set_title("Risk class distribution")
    fig.tight_layout()
    fig.savefig(PLOTS / "class_distribution.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    labels = [config.RISK_LABELS[k] for k in sorted(config.RISK_LABELS)]
    for ax, feature in zip(axes.flat, config.NUMERIC_FEATURES):
        data = [df.loc[df[config.TARGET] == k, feature] for k in sorted(config.RISK_LABELS)]
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(f"{feature} ({config.UNITS[feature]})", fontsize=10)
    axes.flat[-1].axis("off")
    fig.suptitle("Numerical features by risk class")
    fig.tight_layout()
    fig.savefig(PLOTS / "features_by_class.png", dpi=200)
    plt.close(fig)

    corr = df[[*config.NUMERIC_FEATURES, config.TARGET]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation")
    fig.tight_layout()
    fig.savefig(PLOTS / "spearman_correlation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset(config.DATASET_PATH)

    dist = class_distribution(df)
    outliers = iqr_outliers(df)
    feasibility = cv_feasibility(df)

    dist.to_csv(RESULTS / "class_distribution.csv", index=False)
    numeric_summary(df).to_csv(RESULTS / "numeric_summary.csv", index=False)
    numeric_by_class(df).to_csv(RESULTS / "numeric_by_class.csv", index=False)
    df["soil_type"].value_counts().rename_axis("soil_type").reset_index().to_csv(
        RESULTS / "soil_type_distribution.csv", index=False
    )
    pd.crosstab(df["soil_type"], df[config.TARGET].map(config.RISK_LABELS)).reindex(
        columns=list(config.RISK_LABELS.values()), fill_value=0
    ).to_csv(RESULTS / "soil_type_by_risk.csv")
    df[[*config.NUMERIC_FEATURES, config.TARGET]].corr(method="spearman").round(3).to_csv(
        RESULTS / "spearman_correlation.csv"
    )
    outliers.to_csv(RESULTS / "outliers_iqr.csv", index=False)
    df.to_csv(RESULTS / "dataset_as_loaded.csv", index=False)

    overview = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_file": config.DATASET_PATH.name,
        "dataset_sha256": _sha256(config.DATASET_PATH),
        "n_rows": len(df),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "features": config.FEATURES,
        "target": config.TARGET,
        "units": config.UNITS,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_vectors": int(df[config.FEATURES].duplicated().sum()),
        "soil_type_categories": sorted(df["soil_type"].unique().tolist()),
        "cv_feasibility": feasibility,
    }
    (RESULTS / "dataset_overview.json").write_text(json.dumps(overview, indent=2, ensure_ascii=False))

    save_plots(df, dist)

    print(f"Loaded {len(df)} rows from {config.DATASET_PATH.name} (sha256 {overview['dataset_sha256'][:12]}…)")
    print(dist.to_string(index=False))
    print(f"Max stratified folds: {feasibility['max_stratified_folds']}")
    print(f"IQR outlier flags: {len(outliers)} (see outliers_iqr.csv)")
    print(f"Results -> {RESULTS}\nPlots   -> {PLOTS}")


if __name__ == "__main__":
    main()