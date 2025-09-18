# 📊 Customer Churn Prediction  

End-to-End Customer Churn Prediction Project using:  
- **Data Preprocessing** (NumPy, Pandas)  
- **Data Visualization** (Matplotlib, Seaborn)  
- **Machine Learning Models** (Logistic Regression, Random Forest, XGBoost)  
- **Deployment** (Streamlit)  

This project predicts **customer churn** (whether a customer will leave or not) using ML models and provides an interactive web app for real-time prediction.  

---

## 🚀 Project Overview
- Performed **data cleaning & preprocessing** (missing values, encoding, scaling).  
- Trained multiple ML models:  
  - Logistic Regression (ROC-AUC: 0.841)  
  - Random Forest (ROC-AUC: 0.8426)  
  - XGBoost (ROC-AUC: 0.8215)  
- Evaluated models using **Accuracy, Precision, Recall, F1-score, ROC-AUC**.  
- Identified top churn drivers: **Contract type, Tenure, Monthly Charges, Payment Method**.  
- Deployed as an **interactive Streamlit Web App** where users can input customer details and get churn probability.  

---

## 📂 Repository Structure

customer-churn-prediction/

├── WA_Fn-UseC_-Telco-Customer-Churn.csv/ # Dataset

├── Churn-Rate-Analysis.ipynb/ # Jupyter Notebooks (EDA + Modeling)

├── app.py                # Streamlit app

├── rf_churn_model_compressed.pkl  # Trained Random Forest model

├── transformer_compressed.pkl     # Preprocessing transformer (OHE + scaling)

├── requirements.txt      # Dependencies

└── README.md             # Project documentation


---

## 🛠️ Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn, XGBoost  
- Matplotlib, Seaborn  
- Streamlit (for deployment)  

---

## 📊 Model Performance
| Model                | Accuracy | ROC-AUC | Precision | Recall  | F1-Score |
|----------------------|----------|---------|-----------|---------|----------|
| Logistic Regression  | 82%      | 0.841   | 0.6561    | 0.5561  | 0.6020   |
| Random Forest        | 77%      | 0.8426  | 0.5537    | 0.7165  | 0.6247   |
| XGBoost              | 77%      | 0.8215  | 0.5589    | 0.6337  | 0.5940   |

---

## 🎯 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/debugwithmohit/customer-churn-prediction.git
   cd customer-churn-prediction
   
2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

📦 Deployment

The Streamlit App allows users to enter customer details (tenure, monthly charges, contract type, etc.) and predicts churn probability in real-time.

👤 Author

Mohit Kulshreshtha
B.Tech CSE (AI & ML) | Aspiring Data Scientist/ Ml engineer

https://www.linkedin.com/in/mohit-kulshreshtha-47a098290/
