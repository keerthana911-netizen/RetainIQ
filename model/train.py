"""Train, compare and export RetainIQ models.

    python model/train.py [--data path/to/WA_Fn-UseC_-HR-Employee-Attrition.csv]

What it does (in order):
  1. Stratified 80/20 split. The 20% test set is touched ONLY for final reporting.
  2. Builds one identical preprocessing pipeline per model (no leakage: scaling and
     encoding are fitted inside each CV fold).
  3. Handles imbalance (~16% leavers) with class weighting, then ablates it
     (weights on vs off) so the effect is measured, not assumed.
  4. Tunes Logistic Regression, Random Forest and XGBoost with randomized search
     on ROC-AUC, using stratified CV on the training split only.
  5. Reports ROC-AUC, PR-AUC, precision, recall, F1 (not accuracy) with repeated
     CV mean +/- sd AND a held-out test set with bootstrap 95% CIs.
  6. Picks the champion by CV ROC-AUC (never by test score), calibrates its
     probabilities, tunes risk-band thresholds on out-of-fold predictions,
     and exports model + SHAP background + metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import config as C  # noqa: E402
from explain import RiskExplainer, logit, sigmoid  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
DATA_URL = ("https://raw.githubusercontent.com/pplonski/datasets-for-start/refs/heads/"
            "master/employee_attrition/HR-Employee-Attrition-All.csv")
ART = ROOT / "backend" / "artifacts"
IMG = ROOT / "docs" / "img"


# ----------------------------------------------------------------- data
def load_data(path: str | None) -> pd.DataFrame:
    p = Path(path) if path else ROOT / "data" / "HR-Employee-Attrition.csv"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Dataset not found at {p}; downloading public copy of the IBM HR dataset...")
        urllib.request.urlretrieve(DATA_URL, p)
    df = pd.read_csv(p, encoding="utf-8-sig")
    df = df.drop(columns=[c for c in C.DROP_ALWAYS if c in df.columns])
    df[C.TARGET] = (df[C.TARGET] == "Yes").astype(int)
    return df


# ------------------------------------------------------------- pipelines
def make_pre(numeric, categorical):
    return ColumnTransformer(
        [("num", StandardScaler(), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
        sparse_threshold=0,
    )


def make_pipe(clf, numeric, categorical):
    return Pipeline([("pre", make_pre(numeric, categorical)), ("clf", clf)])


def candidates(pos_weight: float):
    """name -> (classifier, param distribution). All prefixed for the Pipeline."""
    lr = LogisticRegression(class_weight="balanced", max_iter=5000, solver="liblinear")
    rf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    xgb = XGBClassifier(scale_pos_weight=pos_weight, eval_metric="logloss", tree_method="hist",
                        n_jobs=-1, random_state=SEED, verbosity=0)
    return {
        "Logistic Regression": (lr, {
            "clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            "clf__penalty": ["l1", "l2"]}, 14),
        "Random Forest": (rf, {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [3, 5, 8, 12, None],
            "clf__min_samples_leaf": [1, 3, 5, 10],
            "clf__max_features": ["sqrt", 0.3, 0.5]}, 20),
        "XGBoost": (xgb, {
            "clf__n_estimators": [150, 300, 500],
            "clf__max_depth": [2, 3, 4, 5],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.4, 0.6, 0.8],
            "clf__min_child_weight": [1, 3, 5],
            "clf__reg_lambda": [1, 5, 10]}, 30),
    }


# --------------------------------------------------------------- metrics
def best_threshold(y, p, beta):
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
    return float(thr[int(np.argmax(f))])


def at_threshold(y, p, t):
    yhat = (p >= t).astype(int)
    return {
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 3),
        "recall": round(float(recall_score(y, yhat)), 3),
        "f1": round(float(f1_score(y, yhat)), 3),
        "flagged_pct": round(float(yhat.mean()) * 100, 1),
    }


def bootstrap_ci(y, p, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
           "precision": "precision", "recall": "recall", "f1": "f1"}


def cv_summary(est, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    r = cross_validate(est, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {k: [round(float(r[f"test_{k}"].mean()), 3), round(float(r[f"test_{k}"].std()), 3)]
            for k in SCORING}


def strip_weights(est):
    """Same tuned model with imbalance weighting switched off (for the ablation)."""
    e = __import__("sklearn.base", fromlist=["clone"]).clone(est)
    clf = e.named_steps["clf"]
    if isinstance(clf, XGBClassifier):
        clf.set_params(scale_pos_weight=1.0)
    else:
        clf.set_params(class_weight=None)
    return e


# ------------------------------------------------------------------ main
def main(data_path):
    t0 = time.time()
    df = load_data(data_path)
    numeric, categorical = C.MODEL_NUMERIC, C.MODEL_CATEGORICAL
    feats = numeric + categorical
    X, y = df[feats], df[C.TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    print(f"rows={len(df)}  leavers={y.mean():.1%}  train={len(X_tr)}  test={len(X_te)}  "
          f"pos_weight={pos_weight:.2f}")

    metrics = {
        "dataset": {"rows": int(len(df)), "attrition_rate": round(float(y.mean()), 4),
                    "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
                    "test_positives": int(y_te.sum()),
                    "source": "IBM HR Analytics Employee Attrition (fictional, synthetic)"},
        "majority_baseline_accuracy": round(float(1 - y_te.mean()), 4),
        "excluded_sensitive_features": C.SENSITIVE,
        "models": {},
    }

    # ---- baseline: always predict "stays" --------------------------------
    dummy = make_pipe(DummyClassifier(strategy="prior"), numeric, categorical).fit(X_tr, y_tr)
    metrics["models"]["Baseline (always 'stays')"] = {
        "test": {"roc_auc": 0.5, "pr_auc": round(float(y_te.mean()), 3),
                 "accuracy": round(float(1 - y_te.mean()), 3), "precision": 0.0, "recall": 0.0, "f1": 0.0},
        "note": "High accuracy, zero leavers caught: why accuracy alone is misleading here.",
    }

    # ---- tune + evaluate each model -------------------------------------
    inner_cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    tuned, oof, results = {}, {}, {}
    for name, (clf, dist, n_iter) in candidates(pos_weight).items():
        t1 = time.time()
        search = RandomizedSearchCV(make_pipe(clf, numeric, categorical), dist, n_iter=n_iter,
                                    scoring="roc_auc", cv=inner_cv, random_state=SEED,
                                    n_jobs=-1, refit=True)
        search.fit(X_tr, y_tr)
        est = search.best_estimator_
        tuned[name] = est
        oof[name] = cross_val_predict(est, X_tr, y_tr, cv=inner_cv, method="predict_proba")[:, 1]
        p_te = est.predict_proba(X_te)[:, 1]
        t_f2, t_f1 = best_threshold(y_tr, oof[name], 2), best_threshold(y_tr, oof[name], 1)
        res = {
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv": cv_summary(est, X_tr, y_tr),
            "test": {
                "roc_auc": round(float(roc_auc_score(y_te, p_te)), 3),
                "roc_auc_ci95": bootstrap_ci(y_te, p_te, roc_auc_score),
                "pr_auc": round(float(average_precision_score(y_te, p_te)), 3),
                "pr_auc_ci95": bootstrap_ci(y_te, p_te, average_precision_score),
                "at_threshold_0.5": at_threshold(y_te, p_te, 0.5),
                "at_recall_oriented_threshold": at_threshold(y_te, p_te, t_f2),
                "at_balanced_threshold": at_threshold(y_te, p_te, t_f1),
            },
            "thresholds_from_oof": {"recall_oriented_F2": round(t_f2, 3), "balanced_F1": round(t_f1, 3)},
        }
        # imbalance ablation: same tuned hyperparameters, weighting OFF
        nw = strip_weights(est)
        nw_cv = cv_summary(nw, X_tr, y_tr)
        nw.fit(X_tr, y_tr)
        nw_te = nw.predict_proba(X_te)[:, 1]
        res["imbalance_ablation"] = {
            "with_class_weight": {"cv_roc_auc": res["cv"]["roc_auc"], "cv_pr_auc": res["cv"]["pr_auc"],
                                  "cv_recall@0.5": res["cv"]["recall"], "cv_precision@0.5": res["cv"]["precision"],
                                  "test@0.5": res["test"]["at_threshold_0.5"]},
            "without_class_weight": {"cv_roc_auc": nw_cv["roc_auc"], "cv_pr_auc": nw_cv["pr_auc"],
                                     "cv_recall@0.5": nw_cv["recall"], "cv_precision@0.5": nw_cv["precision"],
                                     "test@0.5": at_threshold(y_te, nw_te, 0.5)},
        }
        results[name] = res
        metrics["models"][name] = res
        print(f"[{name:20s}] cvAUC={res['cv']['roc_auc'][0]:.3f}  testAUC={res['test']['roc_auc']:.3f}  "
              f"PR-AUC={res['test']['pr_auc']:.3f}  ({time.time()-t1:.0f}s)")

    # ---- champion: by CV ROC-AUC, restricted to models with an exact log-odds SHAP path
    explainable = [n for n in results if n in ("Logistic Regression", "XGBoost")]
    champion = max(explainable, key=lambda n: results[n]["cv"]["roc_auc"][0])
    best_any = max(results, key=lambda n: results[n]["cv"]["roc_auc"][0])
    metrics["champion"] = champion
    metrics["champion_rule"] = ("Highest repeated-CV ROC-AUC among models with exact SHAP log-odds "
                                "attribution (Logistic Regression, XGBoost). "
                                f"Best CV ROC-AUC overall: {best_any}.")
    print("champion:", champion)

    # ---- sensitive-feature ablation (cost of fairness) -------------------
    all_num = C.NUMERIC
    all_cat = C.CATEGORICAL
    champ_est = tuned[champion]
    with_sens = Pipeline([("pre", make_pre(all_num, all_cat)), ("clf", champ_est.named_steps["clf"])])
    cvs = cv_summary(with_sens, df[all_num + all_cat].loc[X_tr.index], y_tr)
    metrics["sensitive_feature_ablation"] = {
        "excluded": C.SENSITIVE,
        "cv_roc_auc_without": results[champion]["cv"]["roc_auc"],
        "cv_roc_auc_with": cvs["roc_auc"],
    }

    # ---- calibrate champion (class weights inflate raw probabilities) ----
    z = logit(oof[champion]).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, solver="lbfgs").fit(z, y_tr)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    calib = {"a": a, "b": b, "method": "Platt scaling on out-of-fold log-odds"}
    cal = lambda p: sigmoid(a * logit(p) + b)  # noqa: E731
    p_te_raw = champ_est.predict_proba(X_te)[:, 1]
    p_te = cal(p_te_raw)
    p_oof = cal(oof[champion])
    metrics["champion_calibration"] = {
        "brier_test_raw": round(float(brier_score_loss(y_te, p_te_raw)), 4),
        "brier_test_calibrated": round(float(brier_score_loss(y_te, p_te)), 4),
        "mean_predicted_test": round(float(p_te.mean()), 4),
        "observed_test_rate": round(float(y_te.mean()), 4),
    }

    # ---- risk bands from OOF operating points ----------------------------
    t_watch = float(cal(best_threshold(y_tr, oof[champion], 2)))   # recall-oriented
    t_high = float(cal(best_threshold(y_tr, oof[champion], 1)))    # balanced
    bands = {"watch": round(t_watch, 4), "high": round(t_high, 4)}

    def band_stats(p, yy):
        out = {}
        for nm, m in (("Low", p < t_watch), ("Medium", (p >= t_watch) & (p < t_high)), ("High", p >= t_high)):
            out[nm] = {"share_of_employees_pct": round(float(m.mean()) * 100, 1),
                       "observed_attrition_pct": round(float(yy[m].mean()) * 100, 1) if m.any() else None}
        return out
    metrics["risk_bands"] = {"thresholds": bands, "oof_train": band_stats(p_oof, y_tr),
                             "test": band_stats(p_te, y_te)}

    # ---- fit champion on all training rows is already done by search.refit --
    # (final model = tuned pipeline refit on the 80% train split; test set stays untouched)
    # SHAP
    bg = X_tr.sample(100, random_state=SEED)
    explainer = RiskExplainer(champ_est, calib, bg, categorical)
    glob = explainer.global_importance(X_tr)
    base_logit, rows = explainer.contributions(X_te.iloc[:5])
    chk = max(abs(sigmoid(base_logit + sum(r.values())) - p)
              for r, p in zip(rows, cal(champ_est.predict_proba(X_te.iloc[:5])[:, 1])))
    assert chk < 1e-3, f"SHAP additivity check failed ({chk})"
    print(f"SHAP additivity check OK (max err {chk:.1e})")

    # ---- defaults / ranges for the API schema -----------------------------
    schema = {"numeric": {}, "categorical": {}}
    for f in numeric:
        s = df[f]
        schema["numeric"][f] = {"min": float(s.min()), "max": float(s.max()),
                                "median": float(s.median()), "integer": bool((s % 1 == 0).all())}
    for f in categorical:
        vc = df[f].value_counts()
        schema["categorical"][f] = {"options": sorted(vc.index.tolist()), "mode": str(vc.index[0])}

    ART.mkdir(parents=True, exist_ok=True)
    meta = {"champion": champion, "features": feats, "numeric": numeric, "categorical": categorical,
            "calibration": calib, "risk_bands": bands, "schema": schema,
            "base_rate": round(float(y.mean()), 4),
            "oof_quantiles": [round(float(q), 5) for q in np.quantile(p_oof, np.linspace(0, 1, 101))]}
    joblib.dump(champ_est, ART / "model.joblib", compress=3)
    joblib.dump(bg, ART / "shap_background.joblib", compress=3)
    json.dump(meta, open(ART / "meta.json", "w"), indent=2)
    json.dump(metrics, open(ART / "metrics.json", "w"), indent=2)
    json.dump(glob, open(ART / "shap_global.json", "w"), indent=2)

    make_figures(results, y_te, tuned, X_te, explainer, champ_est, champion, p_te, y_te)
    print(f"done in {time.time()-t0:.0f}s -> {ART}")


# --------------------------------------------------------------- figures
def make_figures(results, y_te, tuned, X_te, explainer, champ_est, champion, p_te, y):
    IMG.mkdir(parents=True, exist_ok=True)
    colors = {"Logistic Regression": "#4c78a8", "Random Forest": "#59a14f", "XGBoost": "#e15759"}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for n, est in tuned.items():
        p = est.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, color=colors[n], lw=2, label=f"{n} (AUC {roc_auc_score(y_te, p):.2f})")
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[1].plot(rc, pr, color=colors[n], lw=2, label=f"{n} (AP {average_precision_score(y_te, p):.2f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set(title="ROC curve (held-out test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(y_te.mean(), color="k", ls="--", lw=1, label=f"Baseline ({y_te.mean():.2f})")
    ax[1].set(title="Precision-Recall curve (held-out test)", xlabel="Recall", ylabel="Precision")
    for a_ in ax:
        a_.legend(loc="best", fontsize=8); a_.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(IMG / "model_comparison.png", dpi=150); plt.close()

    # SHAP beeswarm on encoded features (champion)
    pre = champ_est.named_steps["pre"]
    Xt = pd.DataFrame(pre.transform(X_te), columns=[c.split("__", 1)[1] for c in pre.get_feature_names_out()])
    sv = explainer._shap_matrix(np.asarray(pre.transform(X_te), dtype=float)) * explainer.a
    plt.figure(figsize=(8.5, 6))
    shap.summary_plot(sv, Xt, max_display=14, show=False, plot_size=None)
    plt.title(f"SHAP summary: {champion} (impact on log-odds of leaving)", fontsize=11)
    plt.tight_layout(); plt.savefig(IMG / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    main(ap.parse_args().data)
