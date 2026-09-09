import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Transmission Line Fault Detection",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("transmission_fault_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df = df.copy()

    # Current magnitude
    df["I_mag"] = np.sqrt(
        df["Ia"]**2 +
        df["Ib"]**2 +
        df["Ic"]**2
    )

    # Voltage magnitude
    df["V_mag"] = np.sqrt(
        df["Va"]**2 +
        df["Vb"]**2 +
        df["Vc"]**2
    )

    # Current imbalance
    df["I_imbalance"] = (
        df[["Ia", "Ib", "Ic"]].max(axis=1)
        - df[["Ia", "Ib", "Ic"]].min(axis=1)
    )

    # Voltage imbalance
    df["V_imbalance"] = (
        df[["Va", "Vb", "Vc"]].max(axis=1)
        - df[["Va", "Vb", "Vc"]].min(axis=1)
    )

    return df


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Section headers */
    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Cards */
    .feature-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        text-align: center;
        margin-bottom: 10px;
    }

    .feature-name {
        font-size: 15px;
        font-weight: 600;
    }

    .feature-value {
        font-size: 25px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Prediction card */
    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(128,128,128,0.3);
        text-align: center;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    .prediction-label {
        font-size: 16px;
        font-weight: 600;
    }

    .prediction-value {
        font-size: 35px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 15px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚡ Transmission Line Fault Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Transmission-Line Fault Classification'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the three-phase current and voltage measurements "
    "using the same scale as the training dataset."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">⚡ Phase Measurements</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# ---------------- CURRENT INPUTS ----------------

with col1:

    st.markdown("### 🔌 Three-Phase Currents")

    Ia = st.number_input(
        "Ia — Phase A Current",
        value=0.0,
        format="%.6f"
    )

    Ib = st.number_input(
        "Ib — Phase B Current",
        value=0.0,
        format="%.6f"
    )

    Ic = st.number_input(
        "Ic — Phase C Current",
        value=0.0,
        format="%.6f"
    )


# ---------------- VOLTAGE INPUTS ----------------

with col2:

    st.markdown("### ⚡ Three-Phase Voltages")

    Va = st.number_input(
        "Va — Phase A Voltage",
        value=0.0,
        format="%.6f"
    )

    Vb = st.number_input(
        "Vb — Phase B Voltage",
        value=0.0,
        format="%.6f"
    )

    Vc = st.number_input(
        "Vc — Phase C Voltage",
        value=0.0,
        format="%.6f"
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "Ia": [Ia],
    "Ib": [Ib],
    "Ic": [Ic],
    "Va": [Va],
    "Vb": [Vb],
    "Vc": [Vc]
})


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("")

if st.button(
    "⚡ ANALYZE FAULT",
    use_container_width=True
):

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    input_features = engineer_features(input_data)

    # Make sure feature order matches training
    input_features = input_features[feature_names]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_features)[0]


    # --------------------------------------------------------
    # Fault descriptions
    # --------------------------------------------------------

    fault_descriptions = {
        "LG": "Single Line-to-Ground Fault",
        "LL": "Line-to-Line Fault",
        "LLG": "Double Line-to-Ground Fault",
        "LLL": "Three-Phase Fault",
        "LLLG": "Three-Phase-to-Ground Fault",
        "Normal": "Normal Operating Condition"
    }


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == "Normal":

        st.success("🟢 NORMAL CONDITION DETECTED")

    else:

        st.error(f"🔴 {prediction} FAULT DETECTED")


    st.markdown("### Classification")

    st.subheader(prediction)

    st.write(fault_descriptions[prediction])


    # ========================================================
    # CALCULATED FEATURES
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Calculated Electrical Features</div>',
        unsafe_allow_html=True
    )


    # Get values
    I_mag = input_features["I_mag"].iloc[0]
    V_mag = input_features["V_mag"].iloc[0]
    I_imbalance = input_features["I_imbalance"].iloc[0]
    V_imbalance = input_features["V_imbalance"].iloc[0]


    # Four columns
   # Four columns
f1, f2, f3, f4 = st.columns(4)

with f1:
    st.metric(
        "Current Magnitude",
        f"{I_mag:.4f}"
    )

with f2:
    st.metric(
        "Voltage Magnitude",
        f"{V_mag:.4f}"
    )

with f3:
    st.metric(
        "Current Imbalance",
        f"{I_imbalance:.4f}"
    )

with f4:
    st.metric(
        "Voltage Imbalance",
        f"{V_imbalance:.4f}"
    )


# ============================================================
# ABOUT SECTION
# ============================================================

st.markdown("---")

st.markdown("## 📘 About This Project")

st.write(
    """
    This application uses a machine learning model to classify
    transmission-line conditions from three-phase voltage and
    current measurements.
    """
)

about_col1, about_col2 = st.columns(2)

with about_col1:

    st.markdown("### Fault Classes")

    st.write(
        """
        - **LG** — Single Line-to-Ground
        - **LL** — Line-to-Line
        - **LLG** — Double Line-to-Ground
        - **LLL** — Three-Phase
        - **LLLG** — Three-Phase-to-Ground
        - **Normal** — Normal operating condition
        """
    )


with about_col2:

    st.markdown("### Machine Learning Pipeline")

    st.write(
        """
        Six electrical measurements are transformed into
        ten features using feature engineering. A tuned
        Decision Tree classifier then predicts the fault class.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ⚡ Transmission Line Fault Detection & Classification
        <br>
        Machine Learning Project | Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)