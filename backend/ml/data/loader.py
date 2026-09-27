import re
import unicodedata
from pathlib import Path

import pandas as pd

from ml import config


class DatasetValidationError(ValueError):
    """Raised when the dataset does not satisfy the expected schema."""


def _normalise_header(text: object) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip().lower()


def _map_headers(columns: list[object]) -> dict[object, str]:
    """Return {original_header: canonical_name} for every recognised column."""
    mapping: dict[object, str] = {}
    for original in columns:
        normalised = _normalise_header(original)
        for canonical, prefixes in config.HEADER_PREFIXES.items():
            if any(normalised.startswith(prefix) for prefix in prefixes):
                if canonical in mapping.values():
                    raise DatasetValidationError(f"Column '{canonical}' matched more than one header.")
                mapping[original] = canonical
                break
    return mapping


def _parse_elevation(value: object) -> float:
    """Accept 678, 678.0, '678m' or '678 m'."""
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*m?\s*", str(value))
    if not match:
        raise DatasetValidationError(f"Cannot parse elevation value: {value!r}")
    return float(match.group(1).replace(",", "."))


def _validate(df: pd.DataFrame) -> None:
    """Collect every schema problem and raise once, so all issues are visible together."""
    problems: list[str] = []
    name = config.ID_COLUMN

    missing = df[[name, *config.FEATURES, config.TARGET]].isna().sum()
    for column, count in missing[missing > 0].items():
        problems.append(f"{column}: {count} missing value(s)")

    for column in config.NUMERIC_FEATURES:
        negative = df[column] < 0
        if negative.any():
            problems.append(f"{column}: negative values for {df.loc[negative, name].tolist()}")

    invalid_risk = ~df[config.TARGET].isin(config.RISK_LABELS.keys())
    if invalid_risk.any():
        problems.append(f"risk: values outside {{0, 1, 2}} for {df.loc[invalid_risk, name].tolist()}")

    duplicated = df[name].duplicated(keep=False)
    if duplicated.any():
        problems.append(f"municipality: duplicated names {sorted(df.loc[duplicated, name])}")

    for low, high in [("min_water_level", "max_water_level"), ("min_slope", "max_slope")]:
        inverted = df[low] > df[high]
        if inverted.any():
            problems.append(f"{low} > {high} for {df.loc[inverted, name].tolist()}")

    if problems:
        raise DatasetValidationError("Dataset validation failed:\n  - " + "\n  - ".join(problems))


def load_dataset(path: Path | str = config.DATASET_PATH) -> pd.DataFrame:
    """Read the Excel file and return a validated DataFrame with canonical column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_excel(path, sheet_name=0)
    mapping = _map_headers(list(raw.columns))

    required = [config.ID_COLUMN, *config.FEATURES, config.TARGET]
    not_found = [c for c in required if c not in mapping.values()]
    if not_found:
        raise DatasetValidationError(
            f"Required column(s) not found: {not_found}. Headers in file: {list(raw.columns)}"
        )

    df = raw[list(mapping)].rename(columns=mapping)
    df = df.dropna(how="all", subset=required).reset_index(drop=True)

    df[config.ID_COLUMN] = df[config.ID_COLUMN].astype(str).str.strip()
    df["soil_type"] = df["soil_type"].astype(str).str.strip()
    df["elevation"] = df["elevation"].map(_parse_elevation)
    for column in config.NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(float)
    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="raise").astype(int)

    _validate(df)

    ordered = [config.ID_COLUMN, *config.FEATURES, config.TARGET]
    ordered += [c for c in config.OPTIONAL_COLUMNS if c in df.columns]
    return df[ordered]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (predictive features only, without municipality) and y."""
    return df[config.FEATURES].copy(), df[config.TARGET].copy()