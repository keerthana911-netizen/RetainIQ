"""RetainIQ API.

  GET  /              service info             GET  /schema      form definition for the UI
  GET  /health        liveness                 GET  /personas    demo employees
  GET  /model-card    metrics, comparison, global SHAP
  POST /predict       risk + SHAP drivers
  POST /whatif        {"employee": {...}, "changes": {...}}  -> before/after risk
  POST /sweep         {"employee": {...}, "feature": "MonthlyIncome"} -> risk curve
"""
import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

import config as C
from predictor import Predictor

log = logging.getLogger("retainiq")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","))

predictor = Predictor()


def _json() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Send a JSON object with Content-Type: application/json.")
    return data


@app.errorhandler(ValueError)
def bad_input(e):
    return jsonify(error=str(e)), 422


@app.errorhandler(413)
def too_big(e):
    return jsonify(error="Request too large."), 413


@app.errorhandler(Exception)
def unexpected(e):
    log.exception("unhandled error")
    return jsonify(error="Internal error."), 500


@app.get("/")
def home():
    return jsonify(service="RetainIQ API", version="2.0", model=predictor.meta["champion"],
                   endpoints=["/health", "/schema", "/personas", "/model-card",
                              "POST /predict", "POST /whatif", "POST /sweep"])


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/schema")
def schema():
    return jsonify(predictor.form_schema())


@app.get("/personas")
def personas():
    return jsonify(C.PERSONAS)


@app.get("/model-card")
def model_card():
    return jsonify(predictor.model_card())


@app.post("/predict")
def predict():
    return jsonify(predictor.predict(_json()))


@app.post("/whatif")
def whatif():
    d = _json()
    return jsonify(predictor.whatif(d.get("employee", {}), d.get("changes")))


@app.post("/sweep")
def sweep():
    d = _json()
    if "feature" not in d:
        raise ValueError("'feature' is required.")
    return jsonify(predictor.sweep(d.get("employee", {}), d["feature"], d.get("values")))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
