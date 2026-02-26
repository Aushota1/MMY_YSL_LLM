"""
Ансамбли классификаторов: VotingClassifier и StackingClassifier для ранга.
"""

from typing import Any, List

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_voting_rf_svc_lr(voting: str = "soft") -> VotingClassifier:
    """Voting: RF + SVC + LogisticRegression."""
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("svc", SVC(kernel="rbf", probability=True)),
        ("lr", LogisticRegression(max_iter=1000)),
    ]
    return VotingClassifier(estimators=estimators, voting=voting)


def get_stacking_rf_svc_xgb() -> StackingClassifier:
    """Stacking: RF, SVC, XGBoost; мета-обучатель — LogisticRegression."""
    if not HAS_XGB:
        raise ImportError("xgboost required for Stacking (RF,SVC,XGB). pip install xgboost")
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("svc", SVC(kernel="rbf", probability=True)),
        ("xgb", xgb.XGBClassifier(eval_metric="logloss")),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
    )


def get_stacking_rf_svc_lr() -> StackingClassifier:
    """Stacking: RF, SVC, LR; мета — LogisticRegression."""
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("svc", SVC(kernel="rbf", probability=True)),
        ("lr", LogisticRegression(max_iter=1000)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
    )


def get_ensemble_model(name: str) -> Any:
    """
    Возвращает модель ансамбля по имени.
    name: "Voting (RF+SVC+LR)", "Stacking (RF,SVC,XGB meta)", "Stacking (RF,SVC,LR)".
    """
    n = (name or "").strip().lower()
    if "voting" in n and "rf" in n:
        return get_voting_rf_svc_lr(voting="soft")
    if "stacking" in n and "xgb" in n:
        return get_stacking_rf_svc_xgb()
    if "stacking" in n:
        return get_stacking_rf_svc_lr()
    return None
