"""Preprocessing for the flood-risk models.

The preprocessor is always placed inside a Pipeline together with the model.
During cross-validation the whole Pipeline is refit on each training fold, so
the scaler's mean/std and the encoder's categories are learned from training
rows only. This is what prevents data leakage.
"""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import config


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Scale numeric features (optional) and one-hot encode soil_type.

    scale_numeric=True  -> for distance/margin-based models (LogReg, SVM, KNN).
    scale_numeric=False -> for tree-based models, which are unaffected by scaling.

    handle_unknown="ignore": a soil type that is absent from a training fold
    (e.g. 'Rendzina', which occurs once) is encoded as all zeros instead of
    raising an error.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, config.NUMERIC_FEATURES),
            (
                "soil_type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model: BaseEstimator, scale_numeric: bool = True) -> Pipeline:
    """Return preprocessing + model as a single estimator (the unit that is evaluated and saved)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )
