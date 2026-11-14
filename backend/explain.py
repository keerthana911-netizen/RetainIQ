"""SHAP explanations for a fitted RetainIQ pipeline.

Design choices
--------------
* Exact, fast explainers (LinearExplainer / TreeExplainer) rather than the slow
  model-agnostic KernelExplainer, so /predict stays well under a second.
* SHAP runs on the encoded matrix; contributions of one-hot columns are then
  summed back onto the *original* feature (JobRole_Sales Executive + JobRole_...
  -> "Job role"), because HR thinks in features, not dummy columns.
* Contributions are reported in log-odds of the *calibrated* risk, i.e. they sum
  exactly to  logit(risk) - logit(base_risk).  `odds_multiplier = exp(contribution)`
  turns that into plain language: "x2.4 the odds of leaving".
"""
from __future__ import annotations

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression

from config import LABELS, SCALE_LABELS


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _origin(col: str, categorical: list[str]) -> str:
    """'num__MonthlyIncome' -> 'MonthlyIncome'; 'cat__JobRole_Sales Executive' -> 'JobRole'."""
    kind, _, rest = col.partition("__")
    if kind == "num":
        return rest
    for feat in sorted(categorical, key=len, reverse=True):  # longest first
        if rest.startswith(feat + "_"):
            return feat
    return rest


def display_value(feature: str, value):
    if feature in SCALE_LABELS:
        try:
            return f"{SCALE_LABELS[feature][int(value)]} ({int(value)})"
        except (KeyError, ValueError, TypeError):
            pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return f"{value:,.0f}" if value == int(value) else f"{value:,.1f}"
    return str(value).replace("_", " ")


class RiskExplainer:
    def __init__(self, pipeline, calibration: dict, background, categorical: list[str]):
        self.pre = pipeline.named_steps["pre"]
        self.clf = pipeline.named_steps["clf"]
        self.a = float(calibration["a"])
        self.b = float(calibration["b"])
        self.categorical = categorical

        Xbg = np.asarray(self.pre.transform(background), dtype=float)
        self.cols = list(self.pre.get_feature_names_out())
        self.origin = [_origin(c, categorical) for c in self.cols]
        self.features = list(dict.fromkeys(self.origin))  # stable order

        if isinstance(self.clf, LogisticRegression):
            self.kind = "linear"
            self.explainer = shap.LinearExplainer(self.clf, Xbg)
        elif type(self.clf).__name__ == "XGBClassifier":
            self.kind = "xgb"
            self.explainer = shap.TreeExplainer(self.clf)  # raw margin = log-odds
        else:
            raise NotImplementedError(
                f"No exact log-odds SHAP path for {type(self.clf).__name__}; "
                "train.py only promotes Logistic Regression or XGBoost to champion."
            )
        ev = np.atleast_1d(self.explainer.expected_value)
        self.base_margin = float(ev[-1])

    # -- core -------------------------------------------------------------
    def _shap_matrix(self, X) -> np.ndarray:
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):  # older shap: one array per class
            sv = sv[-1]
        return np.asarray(sv, dtype=float).reshape(len(X), -1)

    def contributions(self, df):
        """Return (base_logit, per_row list of {feature: calibrated-logit contribution})."""
        X = np.asarray(self.pre.transform(df), dtype=float)
        sv = self._shap_matrix(X) * self.a  # to calibrated log-odds
        base_logit = self.a * self.base_margin + self.b
        rows = []
        for r in sv:
            agg = {f: 0.0 for f in self.features}
            for o, v in zip(self.origin, r):
                agg[o] += v
            rows.append(agg)
        return base_logit, rows

    def explain_row(self, df_row, top_k: int = 8) -> dict:
        base_logit, (agg,) = self.contributions(df_row)
        row = df_row.iloc[0]
        items = []
        for f, c in agg.items():
            items.append({
                "feature": f,
                "label": LABELS.get(f, f),
                "value": display_value(f, row[f]),
                "contribution": round(float(c), 4),
                "odds_multiplier": round(float(np.exp(c)), 3),
                "direction": "increases" if c > 0 else "decreases",
            })
        items.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        total = sum(agg.values())
        shown = items[:top_k]
        rest = items[top_k:]
        return {
            "base_risk": round(float(sigmoid(base_logit)) * 100, 2),
            "final_risk": round(float(sigmoid(base_logit + total)) * 100, 2),
            "units": "log-odds",
            "drivers": shown,
            "other_features": {
                "count": len(rest),
                "contribution": round(float(sum(d["contribution"] for d in rest)), 4),
            },
        }

    def global_importance(self, df) -> list[dict]:
        _, rows = self.contributions(df)
        mean_abs = {f: float(np.mean([abs(r[f]) for r in rows])) for f in self.features}
        out = [{"feature": f, "label": LABELS.get(f, f), "mean_abs_shap": round(v, 4)}
               for f, v in mean_abs.items()]
        return sorted(out, key=lambda d: -d["mean_abs_shap"])
