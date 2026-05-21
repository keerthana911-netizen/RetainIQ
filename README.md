<div align="center">

<pre>
██████╗ ███████╗████████╗ █████╗ ██╗███╗   ██╗██╗ ██████╗ 
██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║████╗  ██║██║██╔═══██╗
██████╔╝█████╗     ██║   ███████║██║██╔██╗ ██║██║██║   ██║
██╔══██╗██╔══╝     ██║   ██╔══██║██║██║╚██╗██║██║██║▄▄ ██║
██║  ██║███████╗   ██║   ██║  ██║██║██║ ╚████║██║╚██████╔╝
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝ ╚══▀▀═╝
</pre>

### *Predict employee attrition before it happens.*

<br/>

![Status](https://img.shields.io/badge/Status-Fullstack%20Prototype-brightgreen?style=for-the-badge)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge&logo=python)
![Frontend](https://img.shields.io/badge/Frontend-React-blue?style=for-the-badge&logo=react)
![Backend](https://img.shields.io/badge/Backend-Flask-black?style=for-the-badge&logo=flask)

<br/>

🔗 **Backend API:** https://retainiq-backend-gmqx.onrender.com/

🎨 **Frontend App:** https://retain-iq-delta.vercel.app/

📦 **Full Repository:** https://github.com/keerthana911-netizen/RetainIQ

📓 **Model Notebook:** https://github.com/keerthana911-netizen/RetainIQ/tree/main/model

</div>

---

## 🔥 Problem Statement

Employee attrition is a major challenge in the IT industry.

Most companies react **after employees leave**, rather than predicting risk early.

**RetainIQ focuses on early prediction and proactive intervention.**

---

## 🧠 What This Project Does

- Predicts employee attrition probability (0–100%)
- Classifies risk level: **High / Medium / Low**
- Provides real-time predictions via API
- Interactive frontend for user input

---

## 🏗️ Architecture

```text
Frontend (React)
        ↓
Fetch API (POST request)
        ↓
Backend (Flask API)
        ↓
ML Model (Logistic Regression)
        ↓
Prediction Response (JSON)
        ↓
Displayed in UI
```

---

## 📊 Model Performance

- Accuracy: **85.26%**
- Model: Logistic Regression (L1 Regularization)
- Dataset: IBM HR Analytics (Kaggle)
- Class Distribution: ~84% stay / ~16% leave

---

## ⚙️ Tech Stack

### 🧠 Machine Learning

- Python
- Pandas
- Scikit-learn

### ⚙️ Backend

- Flask
- REST API (`/predict`)
- JSON handling

### 🎨 Frontend

- React (Vite)
- useState
- Fetch API
- Recharts

---

## 📊 Sample API

### Request

```json
{
  "Age": 30,
  "MonthlyIncome": 4000,
  "OverTime_Yes": 1
}
```

### Response

```json
{
  "risk_score": 52.25,
  "risk_level": "Medium"
}
```

---

## 🚀 Current Features

- [x] ML model trained on HR dataset
- [x] Flask API for predictions
- [x] React frontend UI
- [x] User input form
- [x] Real-time prediction display
- [x] Live deployment on Render
- [x] Live frontend deployment on Vercel

---

## 🚧 In Progress

- [ ] Dashboard analytics
- [ ] SHAP explainability
- [ ] Department-level cohort analytics
- [ ] Authentication system

---

## 📁 Dataset

This project uses the **IBM HR Analytics Employee Attrition & Performance dataset** provided on Kaggle by:

**Fares Sayah**

Dataset Source:

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

---

## 🧠 How It Works

1. User enters employee data
2. Data sent to Flask API
3. Model processes input
4. Returns probability score
5. UI displays result instantly

---

## 💡 Why I Built This

I wanted to build a project that:

- Solves a real-world problem
- Combines ML + full-stack development
- Demonstrates end-to-end system design

---

## 👩‍💻 Author

**Keerthana**  
First-Year CSE Student | ML + Full Stack Learner

🔗 GitHub: https://github.com/keerthana911-netizen

---

<div align="center">

⭐ *Project under active development*

</div>
