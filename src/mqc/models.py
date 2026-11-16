"""Classical models that work on the interpretable QC features."""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_feature_model(name: str, params: dict[str, Any], seed: int) -> Pipeline:
    """Create an untrained scikit-learn pipeline: scale features, then classify.

    Scaling puts all features on a comparable range, which logistic regression
    needs and which makes its coefficients comparable to each other.
    """
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, random_state=seed, **params)
    elif name == "random_forest":
        clf = RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])
