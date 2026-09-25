import streamlit as st
import numpy as np
import pandas as pd
import joblib
import math

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered"
)

# --------------------------------------------------
# Model Loading (Cached for Performance)
# --------------------------------------------------
@st.cache_resource
def load_model(model_path: str = "insurance_model.pkl"):
    return joblib.load(model_path)

try:
    model = load_model()
except Exception:
    st.error(
        "Unable to load 'insurance_model.pkl'. "
        "Please verify the file is in your root directory."
    )
    st.stop()

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
st.title("🏥 Medical Insurance Cost Predictor")
st.write(
    "Enter the individual's details below to estimate "
    "their medical insurance cost."
)
st.divider()

# --------------------------------------------------
# Input Form
# --------------------------------------------------
with st.form("insurance_form"):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=64,
            value=30,
            step=1
        )

        bmi = st.number_input(
            "BMI",
            min_value=15.96,
            max_value=53.15,
            value=25.0,
            step=0.1
        )

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=5,
            value=0,
            step=1
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        smoker = st.selectbox(
            "Smoking Status",
            ["No", "Yes"]
        )

        region = st.selectbox(
            "Region",
            ["Northeast", "Northwest", "Southeast", "Southwest"]
        )

    submitted = st.form_submit_button(
        "🔮 Predict Insurance Cost",
        use_container_width=True
    )

# --------------------------------------------------
# Prediction Processing
# --------------------------------------------------
if submitted:

    # --------------------------------------------------
    # Categorical Encoding
    # --------------------------------------------------
    sex_encoded = 1 if gender == "Male" else 0
    smoker_encoded = 1 if smoker == "Yes" else 0

    region_mapping = {
        "Northeast": 0,
        "Northwest": 1,
        "Southeast": 2,
        "Southwest": 3
    }

    region_encoded = region_mapping[region]

    # --------------------------------------------------
    # Input DataFrame Construction
    # --------------------------------------------------
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex_encoded],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker_encoded],
        "region": [region_encoded]
    })

    # --------------------------------------------------
    # Prediction Execution
    # --------------------------------------------------
    try:
        prediction = float(model.predict(input_data)[0])

    except Exception:
        st.error(
            "An error occurred while generating the prediction."
        )
        st.stop()

    # --------------------------------------------------
    # Prediction Validation
    # --------------------------------------------------

    # Check for NaN or infinite prediction
    if not math.isfinite(prediction):
        st.error(
            "The model returned an invalid prediction. "
            "Please try again."
        )
        st.stop()

    # Handle negative prediction separately
    if prediction < 0:
        st.warning(
            "The trained Linear Regression model produced an "
            "unrealistic negative estimate for this combination "
            "of inputs. Since insurance charges cannot be negative, "
            "a valid insurance cost cannot be displayed for this "
            "input combination."
        )
        st.stop()

    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------
    st.divider()

    st.subheader("Estimated Insurance Cost")

    st.metric(
        label="Predicted Annual Charge",
        value=f"${prediction:,.2f}"
    )

    # --------------------------------------------------
    # Profile Summary
    # --------------------------------------------------
    st.subheader("Profile Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Age",
            "Gender",
            "BMI",
            "Children",
            "Smoker",
            "Region"
        ],
        "Value": [
            age,
            gender,
            f"{bmi:.1f}",
            children,
            smoker,
            region
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Estimated prediction generated by the trained "
        "Machine Learning model."
    )