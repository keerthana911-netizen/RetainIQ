from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))

with open("columns.json", "r") as f:
    columns = json.load(f)

@app.route("/")
def home():
    return "RetainIQ API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    df = df.reindex(columns=columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    return jsonify({
        "risk_score": round(prob * 100, 2),
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
    })
if __name__ == "__main__":
    app.run(debug=True)