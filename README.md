# 🎗️ Breast Cancer Detection & Risk Assessment System

An AI-powered 🧠 machine learning web application that predicts breast cancer classifications using a **Random Forest Classifier**. Built with Python and Streamlit, it provides interactive predictions, CSV data uploads, and downloadable PDF reports.

## ✨ Features

- 🤖 **Machine Learning:** Random Forest classification using the Wisconsin Breast Cancer dataset.
- 📂 **CSV Upload:** Upload patient feature data for analysis.
- ✅ **Input Validation:** Detects missing columns, invalid values, and incomplete data.
- 📊 **Prediction Results:** Displays predicted classifications and model probability estimates.
- 📄 **PDF Reports:** Generate and download prediction reports.
- 🖥️ **Interactive Web App:** Simple, user-friendly interface powered by Streamlit.
- 🧪 **Automated Testing:** Pytest tests help verify model functionality.
- ⚙️ **Continuous Integration:** GitHub Actions automatically runs tests on code changes.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 🌐 Streamlit | Web application interface |
| 🧠 scikit-learn | Machine learning model |
| 🐼 pandas | Data processing |
| 🔢 NumPy | Numerical operations |
| 📦 joblib | Model serialization |
| 📑 ReportLab | PDF report generation |
| 🧪 pytest | Automated testing |
| 🔄 GitHub Actions | Continuous integration |

## 📁 Project Structure

```text
breast-cancer-detection/
├── 📄 app.py
├── 🧠 train_model.py
├── 📦 breast_cancer_model.pkl
├── 📄 create_patient_file.py
├── 📋 requirements.txt
├── 🧪 tests/
│   └── test_app.py
├── 📂 sample_patients/
├── ⚙️ .github/
│   └── workflows/
│       └── tests.yml
└── 📘 README.md
```

## 🚀 Getting Started

### 📌 Prerequisites

- Python installed
- Git installed

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/anjaliyama4/breast-cancer-detection.git
cd breast-cancer-detection
```

### 2️⃣ Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to use the application. 🎉

## 🧪 Run Automated Tests

Install pytest if needed:

```bash
python -m pip install pytest
```

Run the tests:

```bash
python -m pytest -v
```

✅ Automated tests are also configured through GitHub Actions.

## 🧠 Train the Model

To retrain the Random Forest classifier:

```bash
python train_model.py
```

The trained model is saved as `breast_cancer_model.pkl`.

## 📊 Model Evaluation

The training script evaluates the model using classification metrics, including:

- 🎯 Accuracy
- 📈 ROC-AUC
- 📋 Classification report
- 🔍 Confusion matrix

*Run the training script to obtain the current evaluation results. Metrics may vary depending on the dataset and training configuration.*

## 🔮 Future Improvements

- 📈 Add interactive visualizations for prediction results.
- 🧩 Improve test coverage for CSV validation and PDF generation.
- ☁️ Deploy the application online.
- 🎨 Enhance the user interface and accessibility.
- 🔐 Add stronger privacy protections for uploaded patient data.

## ⚠️ Medical Disclaimer

This project is intended **for educational and demonstration purposes only**. It is not a medical device and must not be used to diagnose cancer or make treatment decisions. Predictions may be incorrect and should never replace professional medical advice.

## 👩‍💻 Author

**Anjali Yama**

🔗 [GitHub Profile](https://github.com/anjaliyama4)

---

⭐ If you find this project interesting, consider giving the repository a star!
