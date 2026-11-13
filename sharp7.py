"""SHARP-7 reference implementation; see README.md for unverified choices."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted


FEATURES = [
    "neck_to_height_ratio", "age", "post_preoxygenation_spo2", "snoring",
    "chronic_pain", "ciprofol", "junior_clinician_dyad",
]
NAMES = ["HistGBM", "XGBoost", "TabPFN", "GBM"]


def operating_threshold(y, p, rule):
    kind, target = rule
    if kind == "fixed" and 0 <= target <= 1:
        return float(target)
    if kind not in {"sensitivity", "specificity"} or not 0 < target <= 1:
        raise ValueError("Use ('sensitivity'/'specificity', target) or ('fixed', probability).")
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    thresholds[0] = np.nextafter(1.0, 2.0)
    if kind == "sensitivity":
        return float(thresholds[tpr >= target].max())
    return float(thresholds[1 - fpr >= target].min())


def default_models(seed):
    from tabpfn import TabPFNClassifier
    from xgboost import XGBClassifier

    return [
        HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045,
            max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=2,
            random_state=seed),
        XGBClassifier(n_estimators=280, max_depth=4, learning_rate=0.035,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
            reg_alpha=0.08, reg_lambda=2.5, objective="binary:logistic",
            eval_metric="logloss", n_jobs=1, random_state=seed),
        TabPFNClassifier(device="auto", random_state=seed,
            categorical_features_indices=[3, 4, 5, 6]),
        GradientBoostingClassifier(n_estimators=180, learning_rate=0.035,
            max_depth=2, min_samples_leaf=20, subsample=0.85, random_state=seed),
    ]


class SHARP7:
    def __init__(self, sentinel_rule, confirmatory_rule, seed=20260925, models=None):
        self.sentinel_rule = sentinel_rule
        self.confirmatory_rule = confirmatory_rule
        self.seed = seed
        self.models = models

    @staticmethod
    def _features(X):
        X = X.loc[:, FEATURES].astype(float)
        if np.isinf(X.to_numpy()).any():
            raise ValueError("Infinite feature values are not supported.")
        if not X[FEATURES[3:]].isin([0, 1]).where(X[FEATURES[3:]].notna(), True).all().all():
            raise ValueError("The last four features must be coded 0/1 or missing.")
        return X

    def _splits(self, X, y, groups, seed):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(cv.split(X, y, groups))
        for train, test in splits:
            if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                raise ValueError("Each training and held-out fold must contain both outcomes.")
        return splits

    def fit(self, X, y, groups):
        self.__dict__.pop("thresholds_", None)
        X = self._features(X)
        if X.isna().all().any():
            raise ValueError("Every development feature must have at least one observed value.")
        y, groups = np.asarray(y), np.asarray(groups)
        if y.ndim != 1 or groups.ndim != 1 or len(X) != len(y) or len(y) != len(groups):
            raise ValueError("X, y and groups must be row-aligned and have equal lengths.")
        if set(np.unique(y)) != {0, 1} or pd.isna(groups).any():
            raise ValueError("Use binary outcomes (0/1) and nonmissing patient identifiers.")
        for rule in (self.sentinel_rule, self.confirmatory_rule):
            operating_threshold([0, 1], [0.25, 0.75], rule)
        models = default_models(self.seed) if self.models is None else self.models
        if len(models) != 4:
            raise ValueError("Supply HistGBM, XGBoost, TabPFN and GBM in that order.")
        self.folds_, self.fold_id_ = [], np.empty(len(y), dtype=int)
        self.oof_ = np.empty((len(y), 4))
        for fold, (train, test) in enumerate(self._splits(X, y, groups, self.seed)):
            Xt, yt = X.iloc[train], y[train]
            inner = self._splits(Xt, yt, groups[train], self.seed + fold + 1)
            fitted = []
            for j, estimator in enumerate(models):
                prep = ColumnTransformer([
                    ("continuous", SimpleImputer(strategy="median", keep_empty_features=True), FEATURES[:3]),
                    ("binary", SimpleImputer(strategy="most_frequent", keep_empty_features=True), FEATURES[3:]),
                ])
                model = make_pipeline(prep, clone(estimator))
                if j < 2:
                    model = CalibratedClassifierCV(model, method="sigmoid", cv=inner,
                                                  ensemble=False, n_jobs=1)
                model.fit(Xt, yt)
                self.oof_[test, j] = model.predict_proba(X.iloc[test])[:, 1]
                fitted.append(model)
            self.folds_.append(fitted)
            self.fold_id_[test] = fold + 1
        score = self.oof_[:, :2].mean(axis=1)
        self.thresholds_ = {
            "lower": operating_threshold(y, score, ("sensitivity", 0.75)),
            "upper": operating_threshold(y, score, ("specificity", 0.85)),
            "sentinel": operating_threshold(y, self.oof_[:, 2], self.sentinel_rule),
            "confirmatory": operating_threshold(y, self.oof_[:, 3], self.confirmatory_rule),
        }
        if self.thresholds_["upper"] <= self.thresholds_["lower"]:
            warnings.warn("Upper <= lower: the gray zone is empty; binary classification is unchanged.")
        return self

    def _classify(self, probabilities, index):
        t = self.thresholds_
        score = probabilities[:, :2].mean(axis=1)
        screen = score >= t["lower"]
        rescue = ~screen & (probabilities[:, 2] >= t["sentinel"]) & (probabilities[:, 3] >= t["confirmatory"])
        strata = np.select([screen & (score >= t["upper"]), screen, rescue],
                           ["high_confidence_positive", "gray_zone_positive", "rescued_positive"],
                           default="negative")
        result = pd.DataFrame(probabilities, columns=NAMES, index=index)
        return result.assign(risk_score=score, prediction=(screen | rescue).astype(int), stratum=strata)

    def predict(self, X):
        check_is_fitted(self, "thresholds_")
        X = self._features(X)
        probabilities = np.mean([
            np.column_stack([model.predict_proba(X)[:, 1] for model in fold])
            for fold in self.folds_
        ], axis=0)
        return self._classify(probabilities, X.index)
