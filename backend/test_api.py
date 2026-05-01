import requests

data = {
    "Age": 30,
    "MonthlyIncome": 4000,
    "OverTime_Yes": 1
}

res = requests.post("http://127.0.0.1:5000/predict", json=data)
print(res.json())