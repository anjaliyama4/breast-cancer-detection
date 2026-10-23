# Breast Cancer Detection & Risk Assessment System

## 📌 Project Overview

The **Breast Cancer Detection & Risk Assessment System** is a machine learning project built with Python and Streamlit. It uses a Random Forest classification model trained on the Wisconsin Breast Cancer dataset to classify tumor measurements as **benign or malignant**.

The application provides an interactive interface where users can upload CSV files containing tumor measurements, view model predictions, and generate downloadable PDF reports.

> **Disclaimer:** This project is intended for educational purposes only. It is not a clinically validated medical diagnostic tool and must not be used to make medical decisions.

## 🎯 Objectives

- Apply machine learning techniques to a classification problem.
- Train and evaluate a Random Forest model.
- Build an interactive web interface using Streamlit.
- Process CSV files containing tumor measurements.
- Generate downloadable PDF prediction reports.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** scikit-learn, Random Forest
- **Data Processing:** Pandas, NumPy
- **Web Application:** Streamlit
- **Model Serialization:** Joblib
- **PDF Generation:** ReportLab
- **Version Control:** Git and GitHub

## ✨ Features

- Machine learning model training and evaluation.
- CSV upload for tumor measurement data.
- Benign or malignant classification predictions.
- Prediction confidence display.
- Downloadable PDF reports.
- Sample patient CSV files for testing the application.

## 📂 Project Structure

```text
breast-cancer-detection/
├── app.py
├── train_model.py
├── create_patient_file.py
├── breast_cancer_model.pkl
├── requirements.txt
└── sample_patients/
    ├── patient_001.csv
    ├── patient_002.csv
    └── ...
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/anjaliyama4/breast-cancer-detection.git
```

### 2. Open the project folder

```bash
cd breast-cancer-detection
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application should open in your browser. If it does not, open the local URL shown in the terminal.

## 🧠 Model Training and Evaluation

The `train_model.py` script trains a Random Forest classifier using the Wisconsin Breast Cancer dataset from scikit-learn.

The training script includes:

- Loading the dataset.
- Splitting data into training and testing sets.
- Training a Random Forest classifier.
- Evaluating the model using accuracy, a classification report, and a confusion matrix.
- Saving the trained model as `breast_cancer_model.pkl`.

To train the model again, run:

```bash
python train_model.py
```

Use the actual results from your training run when documenting model performance. No accuracy or other performance values are claimed here.

## 📊 Using the Application

1. Start the Streamlit application.
2. Enter the requested patient information.
3. Upload a CSV file containing the required tumor measurement features.
4. Review the prediction and confidence displayed by the application.
5. Download the generated PDF report if needed.

Use the provided sample CSV files to test the application.

## 🔮 Future Improvements

- Add precision, recall, F1-score, and ROC-AUC visualizations.
- Compare Random Forest with other classification algorithms.
- Add automated unit and integration tests.
- Improve input validation and error handling.
- Add a confusion matrix visualization to the interface.
- Deploy the application for an online demonstration.

## 👩‍💻 Author

**Anjali Yama**

GitHub: [anjaliyama4](https://github.com/anjaliyama4)

## 📄 License

This project is available for educational and learning purposes. Add a license file if you want to specify formal reuse and distribution terms.
