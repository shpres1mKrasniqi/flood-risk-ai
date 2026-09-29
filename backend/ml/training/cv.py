import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

MAX_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42


def n_splits_for(y: pd.Series, max_splits: int = MAX_SPLITS) -> int:
    smallest = int(y.value_counts().min())
    if smallest < 2:
        raise ValueError(f"Smallest class has {smallest} sample(s); stratified CV needs at least 2.")
    return min(max_splits, smallest)


def build_cv(
    y: pd.Series, n_repeats: int = N_REPEATS, random_state: int = RANDOM_STATE
) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=n_splits_for(y), n_repeats=n_repeats, random_state=random_state
    )


def describe_cv(cv: RepeatedStratifiedKFold) -> dict:
    """Settings to store next to the results, so every experiment is reproducible."""
    n_splits = cv.cvargs["n_splits"]
    return {
        "strategy": "RepeatedStratifiedKFold",
        "n_splits": n_splits,
        "n_repeats": cv.n_repeats,
        "total_folds": n_splits * cv.n_repeats,
        "random_state": cv.random_state,
    }
