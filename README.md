# 🎓 Student Performance Prediction using Machine Learning

An AI-powered Student Performance Prediction system built with **Python**, **Scikit-learn**, and **Streamlit**. This project predicts a student's exam score based on academic, personal, and environmental factors using multiple Machine Learning algorithms.

---

## 📌 Project Overview

This project uses a real **Kaggle Student Performance Factors** dataset to predict student exam scores.

The application includes:

- 📊 Data Analysis
- 🧹 Data Preprocessing
- 🤖 Machine Learning Model Training
- 📈 Model Comparison
- 🎯 Exam Score Prediction
- 🌐 Interactive Streamlit Dashboard

---

## 🚀 Features

- Real Kaggle Dataset (6607 Records)
- Data Cleaning & Preprocessing
- Missing Value Handling
- Multiple Machine Learning Models
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Model Performance Comparison
- Interactive Prediction Dashboard
- Download Prediction Report
- Dataset Explorer
- Professional UI using Streamlit

---

## 📂 Project Structure

```text
Student Performance Prediction/
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── models/
│   ├── best_model.pkl
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── model_info.pkl
│   └── model_comparison.csv
│
└── plots/
    ├── actual_vs_predicted_best_model.png
    └── model_comparison_r2.png
```

---

## 📊 Dataset

**Dataset Name**

Student Performance Factors Dataset

Dataset contains:

- 6607 Student Records
- 19 Input Features
- 1 Target Variable (Exam Score)

Target Variable:

```
Exam_Score
```

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

---

## 🤖 Machine Learning Models

The project compares multiple regression models:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best-performing model is automatically saved as:

```
models/best_model.pkl
```

---

## 📈 Model Performance

Evaluation Metrics:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Example:

| Model | R² Score |
|--------|----------|
| Linear Regression | 0.7696 |
| Gradient Boosting | 0.7317 |
| Random Forest | 0.6703 |

---

## 🎯 Input Features

The model uses the following features:

- Hours Studied
- Attendance
- Parental Involvement
- Access to Resources
- Extracurricular Activities
- Sleep Hours
- Previous Scores
- Motivation Level
- Internet Access
- Tutoring Sessions
- Family Income
- Teacher Quality
- School Type
- Peer Influence
- Physical Activity
- Learning Disabilities
- Parental Education Level
- Distance from Home
- Gender

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-prediction.git
```

Move into project folder

```bash
cd student-performance-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train_model.py
```

---

## ▶️ Run Prediction

```bash
python predict.py
```

---

## ▶️ Launch Streamlit Dashboard

```bash
streamlit run app.py
```

---

## 📸 Dashboard

The dashboard includes:

- Dashboard Overview
- Student Prediction Form
- Model Analysis
- Dataset Explorer
- Download Prediction Report

---

## 📊 Workflow

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Model Training
   │
   ▼
Model Comparison
   │
   ▼
Best Model
   │
   ▼
Prediction
   │
   ▼
Streamlit Dashboard
```

---

## 📄 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
```

---

## 🌟 Future Improvements

- Deep Learning Models
- XGBoost
- CatBoost
- Hyperparameter Tuning
- Explainable AI (SHAP)
- Cloud Deployment
- Student Performance Analytics
- PDF Report Generation

---

## 👨‍💻 Author

**Saurabh Kumar**

GitHub:
https://github.com/saurabhkumarik840-spec

---

## ⭐ Support

If you like this project, please ⭐ star the repository on GitHub.

---

## 📜 License

This project is created for educational and learning purposes.