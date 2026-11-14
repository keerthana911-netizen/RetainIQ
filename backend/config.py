"""Single source of truth for features, labels and ranges.

Training (model/train.py), the API (backend/app.py) and the frontend form
(via GET /schema) all read from here, so they can never drift apart.
"""

TARGET = "Attrition"

# Constant / identifier columns in the IBM dataset: carry no signal.
DROP_ALWAYS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

# Protected characteristics. They are EXCLUDED from the model by default:
# an HR tool that scores people partly on gender, age or marital status is a
# legal and ethical liability. train.py reports the AUC cost of excluding them
# (it is small) so the trade-off is visible, not hidden.
SENSITIVE = ["Gender", "MaritalStatus", "Age"]

NUMERIC = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]
CATEGORICAL = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]

ALL_CANDIDATE_FEATURES = NUMERIC + CATEGORICAL
MODEL_FEATURES = [f for f in ALL_CANDIDATE_FEATURES if f not in SENSITIVE]
MODEL_NUMERIC = [f for f in NUMERIC if f not in SENSITIVE]
MODEL_CATEGORICAL = [f for f in CATEGORICAL if f not in SENSITIVE]

# Fields HR will actually think about, in display order, with friendly labels.
# `scale` = 1..N ordinal survey-style answers.
LABELS = {
    "MonthlyIncome": "Monthly income (USD)",
    "OverTime": "Works overtime",
    "JobSatisfaction": "Job satisfaction",
    "EnvironmentSatisfaction": "Environment satisfaction",
    "WorkLifeBalance": "Work-life balance",
    "JobInvolvement": "Job involvement",
    "RelationshipSatisfaction": "Relationship satisfaction",
    "JobLevel": "Job level",
    "JobRole": "Job role",
    "Department": "Department",
    "BusinessTravel": "Business travel",
    "DistanceFromHome": "Distance from home (km)",
    "YearsAtCompany": "Years at company",
    "YearsInCurrentRole": "Years in current role",
    "YearsSinceLastPromotion": "Years since last promotion",
    "YearsWithCurrManager": "Years with current manager",
    "TotalWorkingYears": "Total working years",
    "NumCompaniesWorked": "Companies worked at before",
    "PercentSalaryHike": "Last salary hike (%)",
    "StockOptionLevel": "Stock option level",
    "TrainingTimesLastYear": "Trainings last year",
    "PerformanceRating": "Performance rating",
    "Education": "Education level",
    "EducationField": "Education field",
    "DailyRate": "Daily rate",
    "HourlyRate": "Hourly rate",
    "MonthlyRate": "Monthly rate",
    "Age": "Age",
    "Gender": "Gender",
    "MaritalStatus": "Marital status",
}

SCALE_LABELS = {
    "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very high"},
    "EnvironmentSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very high"},
    "RelationshipSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very high"},
    "JobInvolvement": {1: "Low", 2: "Medium", 3: "High", 4: "Very high"},
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
    "PerformanceRating": {3: "Excellent", 4: "Outstanding"},
    "Education": {1: "Below college", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
}

# Which inputs the UI should show up-front (the rest go under "Advanced").
PRIMARY_FIELDS = [
    "OverTime", "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance",
    "EnvironmentSatisfaction", "JobRole", "Department", "JobLevel",
    "YearsAtCompany", "YearsSinceLastPromotion", "BusinessTravel",
    "DistanceFromHome", "StockOptionLevel", "TotalWorkingYears",
]

# Quick demo personas for the UI (all values inside the training range).
PERSONAS = {
    "Overworked, underpaid junior": {
        "OverTime": "Yes", "MonthlyIncome": 2400, "JobSatisfaction": 1,
        "WorkLifeBalance": 1, "EnvironmentSatisfaction": 1, "JobRole": "Laboratory Technician",
        "Department": "Research & Development", "JobLevel": 1, "YearsAtCompany": 1,
        "YearsSinceLastPromotion": 0, "BusinessTravel": "Travel_Frequently",
        "DistanceFromHome": 22, "StockOptionLevel": 0, "TotalWorkingYears": 2,
        "TrainingTimesLastYear": 1, "YearsInCurrentRole": 1, "YearsWithCurrManager": 0,
    },
    "Settled mid-career engineer": {
        "OverTime": "No", "MonthlyIncome": 7200, "JobSatisfaction": 3,
        "WorkLifeBalance": 3, "EnvironmentSatisfaction": 3, "JobRole": "Research Scientist",
        "Department": "Research & Development", "JobLevel": 2, "YearsAtCompany": 8,
        "YearsSinceLastPromotion": 2, "BusinessTravel": "Travel_Rarely",
        "DistanceFromHome": 5, "StockOptionLevel": 1, "TotalWorkingYears": 12,
        "TrainingTimesLastYear": 3, "YearsInCurrentRole": 5, "YearsWithCurrManager": 5,
    },
    "Stalled sales exec": {
        "OverTime": "Yes", "MonthlyIncome": 4800, "JobSatisfaction": 2,
        "WorkLifeBalance": 2, "EnvironmentSatisfaction": 2, "JobRole": "Sales Executive",
        "Department": "Sales", "JobLevel": 2, "YearsAtCompany": 6,
        "YearsSinceLastPromotion": 5, "BusinessTravel": "Travel_Frequently",
        "DistanceFromHome": 14, "StockOptionLevel": 0, "TotalWorkingYears": 9,
        "TrainingTimesLastYear": 2, "YearsInCurrentRole": 4, "YearsWithCurrManager": 3,
    },
}
