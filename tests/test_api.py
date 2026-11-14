import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import config as C  # noqa: E402
from app import app, predictor  # noqa: E402
from explain import logit, sigmoid  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return app.test_client()


P = C.PERSONAS


def post(client, path, body):
    return client.post(path, json=body)


def test_health_and_schema(client):
    assert client.get("/health").json == {"status": "ok"}
    s = client.get("/schema").json
    names = {f["name"] for f in s["fields"]}
    assert "OverTime" in names and "MonthlyIncome" in names
    # protected attributes must never be model inputs
    assert not names & set(C.SENSITIVE)


def test_personas_rank_sensibly(client):
    risk = {n: post(client, "/predict", e).json["risk_score"] for n, e in P.items()}
    assert risk["Overworked, underpaid junior"] > risk["Stalled sales exec"] > risk["Settled mid-career engineer"]


def test_levels_follow_bands(client):
    r = post(client, "/predict", P["Settled mid-career engineer"]).json
    assert r["risk_level"] == "Low"
    r = post(client, "/predict", P["Overworked, underpaid junior"]).json
    assert r["risk_level"] == "High"


def test_shap_is_additive_in_calibrated_log_odds(client):
    """base + sum(all feature contributions) must reproduce the displayed risk."""
    df, _ = predictor.complete(P["Stalled sales exec"])
    base_logit, (agg,) = predictor.explainer.contributions(df)
    assert math.isclose(sigmoid(base_logit + sum(agg.values())), predictor.score(df), abs_tol=1e-6)


def test_explanation_shape(client):
    r = post(client, "/predict", P["Stalled sales exec"]).json
    d = r["explanation"]["drivers"]
    assert 1 <= len(d) <= 8
    assert all(abs(d[i]["contribution"]) >= abs(d[i + 1]["contribution"]) for i in range(len(d) - 1))
    assert d[0]["feature"] == "OverTime" and d[0]["direction"] == "increases"


def test_missing_fields_are_reported_not_hidden(client):
    r = post(client, "/predict", {"OverTime": "Yes"}).json
    assert "MonthlyIncome" in r["assumed_defaults"]


def test_out_of_range_is_clipped_and_reported(client):
    r = post(client, "/predict", {"MonthlyIncome": 90000}).json
    assert r["clipped_inputs"][0]["used"] == 19999


def test_sensitive_fields_ignored(client):
    a = post(client, "/predict", {**P["Stalled sales exec"], "Gender": "Male", "Age": 22}).json
    b = post(client, "/predict", P["Stalled sales exec"]).json
    assert a["risk_score"] == b["risk_score"]
    assert set(a["ignored_fields"]) == {"Gender", "Age"}


def test_legacy_v1_payload_still_works(client):
    r = post(client, "/predict", {"Age": 30, "MonthlyIncome": 4000, "OverTime_Yes": 1})
    assert r.status_code == 200 and r.json["inputs_used"]["OverTime"] == "Yes"


@pytest.mark.parametrize("bad", [{"OverTime": "Maybe"}, {"MonthlyIncome": "lots"}, {"JobLevel": True}])
def test_bad_input_is_422(client, bad):
    assert post(client, "/predict", bad).status_code == 422


def test_non_json_is_422(client):
    assert client.post("/predict", data="nope").status_code == 422


def test_whatif_overtime_off_lowers_risk(client):
    e = P["Stalled sales exec"]
    r = post(client, "/whatif", {"employee": e, "changes": {"OverTime": "No"}}).json
    assert r["delta_pp"] < -10
    assert r["scenario"]["risk_score"] < r["baseline"]["risk_score"]
    assert r["changed"][0]["from"] == "Yes" and r["changed"][0]["to"] == "No"
    assert "caveat" in r


def test_whatif_rejects_unknown_and_empty(client):
    e = P["Stalled sales exec"]
    assert post(client, "/whatif", {"employee": e, "changes": {"Salary": 1}}).status_code == 422
    assert post(client, "/whatif", {"employee": e, "changes": {}}).status_code == 422


def test_sweep_income_returns_curve(client):
    r = post(client, "/sweep", {"employee": P["Stalled sales exec"], "feature": "MonthlyIncome"}).json
    assert len(r["points"]) == 25
    assert r["points"][0]["risk_score"] > r["points"][-1]["risk_score"]  # more pay -> lower risk


def test_model_card_has_proper_metrics(client):
    m = client.get("/model-card").json["metrics"]
    assert set(["Logistic Regression", "Random Forest", "XGBoost"]) <= set(m["models"])
    t = m["models"][m["champion"]]["test"]
    assert {"roc_auc", "pr_auc"} <= set(t) and "precision" in t["at_balanced_threshold"]
