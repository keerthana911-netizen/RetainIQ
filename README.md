<div align="center">

```
██████╗ ███████╗████████╗ █████╗ ██╗███╗   ██╗██╗ ██████╗ 
██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║████╗  ██║██║██╔═══██╗
██████╔╝█████╗     ██║   ███████║██║██╔██╗ ██║██║██║   ██║
██╔══██╗██╔══╝     ██║   ██╔══██║██║██║╚██╗██║██║██║▄▄ ██║
██║  ██║███████╗   ██║   ██║  ██║██║██║ ╚████║██║╚██████╔╝
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝ ╚══▀▀═╝
```

### *Predict employee attrition before it happens.*

<br/>

![Status](https://img.shields.io/badge/Status-Fullstack%20Prototype-brightgreen?style=for-the-badge)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge\&logo=python)
![Frontend](https://img.shields.io/badge/Frontend-React-blue?style=for-the-badge\&logo=react)
![Backend](https://img.shields.io/badge/Backend-Flask-black?style=for-the-badge\&logo=flask)

<br/>

🔗 **Backend API:** https://github.com/keerthana911-netizen/RetainIQ/tree/main/backend  
🎨 **Frontend App:** https://github.com/keerthana911-netizen/RetainIQ/tree/main/frontend  
📦 **Full Repository:** https://github.com/keerthana911-netizen/RetainIQ

</div>

---

## 🔥 Problem Statement

Employee attrition is a major challenge in the IT industry.
Companies often react *after* employees leave — not before.

**RetainIQ focuses on early prediction.**

---

## 🧠 What This Project Does

* Predicts employee attrition probability (0–100%)
* Classifies risk level: **High / Medium / Low**
* Provides real-time predictions via API
* Interactive frontend for user input

---

## 🏗️ Architecture

```
Frontend (React)
        ↓
Fetch API (POST request)
        ↓
Backend (Flask API)
        ↓
ML Model (Random Forest)
        ↓
Prediction Response (JSON)
        ↓
Displayed in UI
```

---

## ⚙️ Tech Stack

### 🧠 Machine Learning

* Python
* Pandas
* Scikit-learn (Random Forest)

### ⚙️ Backend

* Flask
* REST API (`/predict`)
* JSON handling

### 🎨 Frontend

* React (Vite)
* useState (state management)
* Fetch API

---

## 📊 Sample API

### Request:

```json
{
  "Age": 30,
  "MonthlyIncome": 4000,
  "OverTime_Yes": 1
}
```

### Response:

```json
{
  "risk_score": 52.25,
  "risk_level": "Medium"
}
```

---

## 🚀 Current Features

* [x] ML model trained on IBM HR dataset
* [x] Flask API for predictions
* [x] React frontend UI
* [x] User input form
* [x] Real-time prediction display

---

## 🚧 In Progress

* [ ] Dashboard UI (charts & analytics)
* [ ] Model explainability (why at risk)
* [ ] Deployment (Render + Vercel)
* [ ] Improved UI/UX

---

## 🧠 How It Works

1. User enters employee details
2. Data sent to Flask API
3. Model processes input
4. Returns probability
5. UI displays result instantly

---

## 💡 Why I Built This

I wanted to build a project that:

* Solves a real-world problem
* Combines ML + full-stack development
* Demonstrates end-to-end system design

---

## 👩‍💻 Author

**Keerthana**
First-Year CSE Student | ML + Full Stack Learner

🔗 GitHub: YOUR_GITHUB_LINK

---

<div align="center">

⭐ *Project under active development*

</div>
