
import os
import html
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Detection System",
    page_icon="🏥",
    layout="wide",
)


# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("breast_cancer_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()


# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🏥 Breast Cancer Detection & Risk Assessment System")

st.markdown(
    """
    This application uses a machine-learning model trained on the
    Wisconsin Breast Cancer Dataset to classify tumor measurements.
    
    **Educational project only:** Predictions are not medical diagnoses.
    """
)

st.divider()


# -----------------------------
# PATIENT INFORMATION
# -----------------------------
st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")

with col2:
    patient_age = st.number_input(
        "Patient Age",
        min_value=1,
        max_value=120,
        value=25,
        step=1,
    )

st.divider()


# -----------------------------
# CSV UPLOAD
# -----------------------------
st.subheader("📂 Upload Patient Measurement File")

uploaded_file = st.file_uploader(
    "Choose Patient CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        patient_data = pd.read_csv(uploaded_file)

        if patient_data.empty:
            st.error("The uploaded CSV file is empty.")
            st.stop()

        # Validate the input against the model's expected features
        expected_features = getattr(
            model, "feature_names_in_", None
        )

        if expected_features is not None:
            missing_columns = [
                col for col in expected_features
                if col not in patient_data.columns
            ]

            extra_columns = [
                col for col in patient_data.columns
                if col not in expected_features
            ]

            if missing_columns:
                st.error(
                    "CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )
                st.stop()

            if extra_columns:
                st.error(
                    "CSV contains unexpected columns: "
                    + ", ".join(extra_columns)
                )
                st.stop()

            # Keep columns in the exact training order
            patient_data = patient_data[list(expected_features)]

        # Ensure all input features are numeric
        patient_data = patient_data.apply(
            pd.to_numeric, errors="coerce"
        )

        if patient_data.isnull().values.any():
            st.error(
                "The CSV contains missing or non-numeric values. "
                "Please check the file and upload it again."
            )
            st.stop()

        if not np.isfinite(patient_data.to_numpy()).all():
            st.error("The CSV contains infinite or invalid values.")
            st.stop()

        st.success("✅ File uploaded and validated successfully.")

        st.subheader("📋 Uploaded Measurements")

        preview = patient_data.T.reset_index()
        preview.columns = ["Feature", "Value"]

        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()


# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:
    st.divider()

    if st.button("🔍 Predict", type="primary"):
        try:
            prediction = model.predict(patient_data)[0]

            # Map the prediction using the model's actual class labels
            classes = list(model.classes_)
            probability = model.predict_proba(patient_data)[0]

            predicted_index = classes.index(prediction)
            confidence = float(probability[predicted_index]) * 100

            # sklearn's built-in breast cancer dataset uses:
            # 0 = malignant, 1 = benign
            if prediction == 1:
                cancer_status = "Benign"
                risk_level = "Lower Model-Indicated Risk"
                recommendation = (
                    "The model classified the measurement as benign. "
                    "This result cannot rule out cancer. Consult a "
                    "qualified healthcare professional for interpretation."
                )
            else:
                cancer_status = "Malignant"
                risk_level = "Higher Model-Indicated Risk"
                recommendation = (
                    "The model classified the measurement as malignant. "
                    "This is not a confirmed diagnosis. Please seek prompt "
                    "evaluation from a qualified healthcare professional."
                )

            # -----------------------------
            # DISPLAY RESULTS
            # -----------------------------
            st.subheader("🩺 Prediction Result")

            if cancer_status == "Benign":
                st.success(f"Prediction: {cancer_status}")
            else:
                st.error(f"Prediction: {cancer_status}")

            st.info(
                f"📊 Model Confidence Score: {confidence:.2f}%"
            )

            st.warning(f"Risk Category: {risk_level}")
            st.progress(min(max(confidence / 100, 0.0), 1.0))

            st.subheader("💡 Recommendation")
            st.write(recommendation)

            st.caption(
                "This educational model is not intended for clinical "
                "decision-making. Its confidence score is not necessarily "
                "a calibrated probability of disease."
            )

            # -----------------------------
            # GENERATE PDF REPORT
            # -----------------------------
            os.makedirs("reports", exist_ok=True)

            safe_name = "".join(
                character
                for character in patient_name.strip()
                if character.isalnum() or character in (" ", "-", "_")
            ).strip().replace(" ", "_")

            if not safe_name:
                safe_name = "Patient"

            pdf_file = os.path.join(
                "reports",
                f"{safe_name}_Report.pdf",
            )

            styles = getSampleStyleSheet()
            elements = []

            elements.append(
                Paragraph(
                    "BREAST CANCER DETECTION REPORT",
                    styles["Title"],
                )
            )
            elements.append(Spacer(1, 16))

            elements.append(
                Paragraph(
                    f"<b>Patient Name:</b> {html.escape(patient_name.strip() or 'Not provided')}",
                    styles["Normal"],
                )
            )
            elements.append(
                Paragraph(
                    f"<b>Patient Age:</b> {patient_age}",
                    styles["Normal"],
                )
            )
            elements.append(
                Paragraph(
                    f"<b>Date:</b> {datetime.now().strftime('%d-%m-%Y %I:%M %p')}",
                    styles["Normal"],
                )
            )

            elements.append(Spacer(1, 16))

            elements.append(
                Paragraph(
                    f"<b>Model Prediction:</b> {cancer_status}",
                    styles["Normal"],
                )
            )
            elements.append(
                Paragraph(
                    f"<b>Risk Category:</b> {risk_level}",
                    styles["Normal"],
                )
            )
            elements.append(
                Paragraph(
                    f"<b>Model Confidence Score:</b> {confidence:.2f}%",
                    styles["Normal"],
                )
            )

            elements.append(Spacer(1, 16))
            elements.append(
                Paragraph(
                    html.escape(recommendation),
                    styles["Normal"],
                )
            )

            elements.append(Spacer(1, 16))
            elements.append(
                Paragraph(
                    "<i>Educational machine-learning output only. "
                    "This report is not a medical diagnosis. "
                    "Consult a qualified healthcare professional.</i>",
                    styles["Normal"],
                )
            )

            doc = SimpleDocTemplate(pdf_file)
            doc.build(elements)

            # -----------------------------
            # PDF DOWNLOAD
            # -----------------------------
            with open(pdf_file, "rb") as file:
                pdf_bytes = file.read()

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_file),
                mime="application/pdf",
            )

        except Exception as e:
            st.error(f"Prediction or report generation error: {e}")
