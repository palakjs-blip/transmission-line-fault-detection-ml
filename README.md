# ⚡ Transmission Line Fault Detection and Classification using Machine Learning

A machine learning project that detects and classifies transmission line operating conditions and electrical faults using three-phase current and voltage measurements.

## 📌 Project Overview

Transmission line faults can cause serious problems in electrical power systems. Early detection and classification of faults can help improve the reliability and protection of power systems.

This project uses machine learning to classify six different transmission line conditions from three-phase electrical measurements.

### Fault Classes

- Normal — Normal Operating Condition
- LG — Single Line-to-Ground Fault
- LL — Line-to-Line Fault
- LLG — Double Line-to-Ground Fault
- LLL — Three-Phase Fault
- LLLG — Three-Phase-to-Ground Fault

## 📊 Dataset

The project uses the **Electrical Fault Detection and Classification** dataset from Kaggle.

The dataset contains three-phase current and voltage measurements along with fault-condition labels.

Input measurements:

- Ia — Phase A Current
- Ib — Phase B Current
- Ic — Phase C Current
- Va — Phase A Voltage
- Vb — Phase B Voltage
- Vc — Phase C Voltage

The dataset contains **7,861 samples**.

Dataset source:

https://www.kaggle.com/datasets/esathyaprakash/electrical-fault-detection-and-classification

## 🛠️ Feature Engineering

In addition to the original six electrical measurements, four additional features were created:

- **Current Magnitude:** `I_mag`
- **Voltage Magnitude:** `V_mag`
- **Current Imbalance:** `I_imbalance`
- **Voltage Imbalance:** `V_imbalance`

These features provide additional information about the overall magnitude and imbalance of the three-phase electrical quantities.

## 🤖 Machine Learning Models

The following classification models were evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Feature engineering significantly improved the performance of the Decision Tree model.

The final model uses a tuned **Decision Tree Classifier**.

## 📈 Results

The final Decision Tree model achieved approximately:

- **Test Accuracy:** 95.8%
- **Best Cross-Validation Accuracy:** 95.28%

The model performed particularly well on most fault classes, while **LLL and LLLG** were relatively more difficult to distinguish.

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 🚀 Streamlit Application

A Streamlit web application was developed to allow users to enter three-phase current and voltage measurements and receive a predicted operating condition or fault classification.

### Run Locally

Install the required packages:

```bash
pip install -r requirements.txt

The application expects input values on the same scale as the training dataset.

📁 Project Structure
transmission-line-fault-detection-ml/
│
├── app.py
├── transmission_fault_model.pkl
├── feature_names.pkl
├── requirements.txt
├── README.md
└── .gitignore

⚠️ Limitations

This model was trained using a simulated electrical power-system dataset. Therefore, its performance on real-world transmission systems may differ.

The model should be used with measurements following the same scaling and conditions represented in the training data.

👩‍💻 Project

This project was developed as a machine learning and electrical engineering application combining power-system concepts with data science.