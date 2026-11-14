"""Inference layer: validation -> calibrated risk -> SHAP drivers -> what-if."""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as C
from explain import RiskExplainer, logit, sigmoid

ART = Path(__file__).resolve().parent / "artifacts"


class Predictor:
    def __init__(self, art_dir: Path = ART):
        self.meta = json.loads((art_dir / "meta.json").read_text())
        self.metrics = json.loads((art_dir / "metrics.json").read_text())
        self.global_shap = json.loads((art_dir / "shap_global.json").read_text())
        self.model = joblib.load(art_dir / "model.joblib")
        bg = joblib.load(art_dir / "shap_background.joblib")
        self.features = self.meta["features"]
        self.numeric, self.categorical = self.meta["numeric"], self.meta["categorical"]
        self.schema = self.meta["schema"]
        self.a, self.b = self.meta["calibration"]["a"], self.meta["calibration"]["b"]
        self.bands = self.meta["risk_bands"]
        self.q = np.array(self.meta["oof_quantiles"])
        self.explainer = RiskExplainer(self.model, self.meta["calibration"], bg, self.categorical)

    # ---------------------------------------------------------- validation
    def complete(self, payload: dict) -> tuple[pd.DataFrame, dict]:
        """Validate an employee record; fill gaps with training medians/modes (and say so)."""
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object of employee fields.")
        payload = dict(payload)
        if "OverTime_Yes" in payload and "OverTime" not in payload:  # legacy v1 API
            payload["OverTime"] = "Yes" if payload.pop("OverTime_Yes") in (1, True, "1", "Yes") else "No"

        row, assumed, clipped = {}, [], []
        for f in self.numeric:
            spec = self.schema["numeric"][f]
            if f in payload and payload[f] is not None and payload[f] != "":
                v = payload[f]
                if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                    raise ValueError(f"'{f}' must be a number.")
                try:
                    v = float(v)
                except ValueError:
                    raise ValueError(f"'{f}' must be a number, got {payload[f]!r}.") from None
                if not math.isfinite(v):
                    raise ValueError(f"'{f}' must be a finite number.")
                if v < spec["min"] or v > spec["max"]:
                    clipped.append({"feature": f, "given": v, "used": min(max(v, spec["min"]), spec["max"])})
                    v = min(max(v, spec["min"]), spec["max"])
                row[f] = v
            else:
                row[f] = spec["median"]
                assumed.append(f)
        for f in self.categorical:
            spec = self.schema["categorical"][f]
            if f in payload and payload[f] not in (None, ""):
                if payload[f] not in spec["options"]:
                    raise ValueError(f"'{f}' must be one of {spec['options']}, got {payload[f]!r}.")
                row[f] = payload[f]
            else:
                row[f] = spec["mode"]
                assumed.append(f)
        ignored = [k for k in payload if k not in self.features]
        return pd.DataFrame([row])[self.features], {"assumed": assumed, "clipped": clipped, "ignored": ignored}

    # ------------------------------------------------------------- scoring
    def score(self, df: pd.DataFrame) -> float:
        raw = float(self.model.predict_proba(df)[0, 1])
        return float(sigmoid(self.a * logit(raw) + self.b))

    def level(self, p: float) -> str:
        return "High" if p >= self.bands["high"] else "Medium" if p >= self.bands["watch"] else "Low"

    def percentile(self, p: float) -> float:
        """Share of (out-of-fold) employees whose predicted risk is lower than p."""
        return float(np.interp(p, self.q, np.linspace(0, 100, len(self.q))))

    def _summary(self, p: float) -> dict:
        return {"risk_score": round(p * 100, 1), "risk_level": self.level(p),
                "riskier_than_pct": round(self.percentile(p), 0)}

    # ----------------------------------------------------------------- API
    def predict(self, payload: dict, top_k: int = 8) -> dict:
        df, notes = self.complete(payload)
        p = self.score(df)
        expl = self.explainer.explain_row(df, top_k=top_k)
        return {**self._summary(p), "explanation": expl,
                "inputs_used": {k: _py(v) for k, v in df.iloc[0].items()},
                "assumed_defaults": notes["assumed"], "clipped_inputs": notes["clipped"],
                "ignored_fields": notes["ignored"],
                "model": {"name": self.meta["champion"], "base_rate_pct": round(self.meta["base_rate"] * 100, 1)}}

    def whatif(self, payload: dict, changes: dict, top_k: int = 5) -> dict:
        if not isinstance(changes, dict) or not changes:
            raise ValueError("'changes' must be a non-empty object, e.g. {\"OverTime\": \"No\"}.")
        bad = [k for k in changes if k not in self.features]
        if bad:
            raise ValueError(f"Cannot change {bad}: not a model input. Valid: {self.features}")
        base_df, n1 = self.complete(payload)
        scen_df, n2 = self.complete({**payload, **changes})
        p0, p1 = self.score(base_df), self.score(scen_df)
        _, (c0,) = self.explainer.contributions(base_df)
        _, (c1,) = self.explainer.contributions(scen_df)
        shifts = sorted(
            ({"feature": f, "label": C.LABELS.get(f, f), "change_in_contribution": round(c1[f] - c0[f], 4)}
             for f in c0 if abs(c1[f] - c0[f]) > 1e-6),
            key=lambda d: -abs(d["change_in_contribution"]))[:top_k]
        return {
            "baseline": self._summary(p0), "scenario": self._summary(p1),
            "delta_pp": round((p1 - p0) * 100, 1),
            "changed": [{"feature": k, "label": C.LABELS.get(k, k),
                         "from": _py(base_df.iloc[0][k]), "to": _py(scen_df.iloc[0][k])} for k in changes],
            "driver_shifts": shifts, "clipped_inputs": n2["clipped"],
            "caveat": ("The model learned correlations in historical data. A change in predicted risk "
                       "shows what the model associates with different inputs, not a guaranteed causal "
                       "effect of making that change."),
        }

    def sweep(self, payload: dict, feature: str, values: list | None = None, n: int = 25) -> dict:
        if feature not in self.features:
            raise ValueError(f"'{feature}' is not a model input.")
        df, _ = self.complete(payload)
        if feature in self.numeric:
            spec = self.schema["numeric"][feature]
            vals = values or [round(v, 2) for v in np.linspace(spec["min"], spec["max"], n)]
        else:
            vals = values or self.schema["categorical"][feature]["options"]
        pts = []
        for v in vals:
            d2, notes = self.complete({**df.iloc[0].to_dict(), feature: v})
            pts.append({"value": _py(d2.iloc[0][feature]), "risk_score": round(self.score(d2) * 100, 1)})
        return {"feature": feature, "label": C.LABELS.get(feature, feature),
                "current": _py(df.iloc[0][feature]), "points": pts}

    # -------------------------------------------------------------- schema
    def form_schema(self) -> dict:
        fields = []
        for f in self.features:
            base = {"name": f, "label": C.LABELS.get(f, f), "primary": f in C.PRIMARY_FIELDS}
            if f in self.numeric:
                s = self.schema["numeric"][f]
                kind = "scale" if f in C.SCALE_LABELS else "number"
                base.update(type=kind, min=s["min"], max=s["max"], default=s["median"],
                            step=1 if s["integer"] else 0.1)
                if kind == "scale":
                    base["options"] = [{"value": k, "label": v} for k, v in C.SCALE_LABELS[f].items()]
            else:
                s = self.schema["categorical"][f]
                base.update(type="select", default=s["mode"],
                            options=[{"value": o, "label": o.replace("_", " ")} for o in s["options"]])
            fields.append(base)
        fields.sort(key=lambda d: (not d["primary"], C.PRIMARY_FIELDS.index(d["name"])
                                   if d["name"] in C.PRIMARY_FIELDS else 99, d["label"]))
        return {"fields": fields, "excluded_for_fairness": C.SENSITIVE}

    def model_card(self) -> dict:
        return {"metrics": self.metrics, "global_importance": self.global_shap[:12],
                "risk_bands": self.bands}


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if float(v) != int(v) else int(v)
    return v
