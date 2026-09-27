import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CUSTOM DESIGN

st.markdown("""
<style>

/* ---------- FULL BACKGROUND ---------- */

.stApp {
    background: linear-gradient(135deg, #080808, #111111, #080808) !important;
    color: white !important;
}

section[data-testid="stMain"] {
    background: #080808 !important;
}

header[data-testid="stHeader"] {
    background: #080808 !important;
}

[data-testid="stToolbar"] {
    background: #080808 !important;
}


/* ---------- MAIN CONTAINER ---------- */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ---------- TITLE ---------- */

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
    letter-spacing: 1px;
}

.main-title span {
    color: #ff3b4f;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    font-size: 17px;
    margin-bottom: 35px;
}


/* ---------- TOP HEART ---------- */

.heart-icon {
    text-align: center;
    font-size: 55px;
    margin-bottom: 5px;
}


/* ---------- CARDS ---------- */

.card {
    background: linear-gradient(145deg, #1b1b1b, #111111);
    border: 1px solid #2d2d2d;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
}

.card-title {
    font-size: 23px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 20px;
}


/* ---------- INPUT LABELS ---------- */

label {
    color: #eeeeee !important;
    font-weight: 500 !important;
}


/* ---------- INPUT BOXES ---------- */

div[data-baseweb="select"] > div {
    background-color: #202020 !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 10px !important;
}

input {
    background-color: #202020 !important;
    color: white !important;
    border-radius: 10px !important;
}


/* ---------- PREDICT BUTTON ---------- */

div.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(90deg, #e51c3b, #ff4056);
    color: white;
    font-size: 21px;
    font-weight: 700;
    box-shadow: 0 6px 20px rgba(255, 45, 75, 0.25);
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(90deg, #ff4056, #e51c3b);
    box-shadow: 0 8px 25px rgba(255, 45, 75, 0.4);
}


/* ---------- RESULT TITLE ---------- */

.result-title {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #777777;
    font-size: 13px;
    margin-top: 40px;
}


/* ---------- DIVIDER ---------- */

hr {
    border-color: #292929 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="heart-icon">❤️</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Heart Disease <span>Prediction</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Heart Disease Risk Prediction System'
    '</div>',
    unsafe_allow_html=True
)

# PATIENT INFORMATION

st.markdown("""
<div class="card">
<div class="card-title">👤 Patient Information</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)


with col1:

    age = st.slider(
        "Age",
        18,
        100,
        40
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        80,
        200,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        100,
        600,
        200
    )


with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

st.markdown("</div>", unsafe_allow_html=True)

# HEART PARAMETERS

st.markdown("""
<div class="card">
<div class="card-title">❤️ Heart Health Parameters</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)


with col1:

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )


with col2:

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )


with col3:

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# PREDICT BUTTON

st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "🔍  PREDICT HEART DISEASE",
    use_container_width=True
):

    # RAW INPUT
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex": 1,
        "ChestPainType": chest_pain,
        "RestingECG": resting_ecg,
        "ExerciseAngina": exercise_angina,
        "ST_Slope": st_slope
    }


      # DATAFRAME

    input_df = pd.DataFrame([raw_input])

  # ADD MISSING COLUMNS
    
    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col] = 0

    # COLUMN ORDER
    input_df = input_df[expected_columns]

    # SCALE INPUT
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    # RESULT
    
    st.markdown(
        '<div class="result-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK OF HEART DISEASE"
        )

    else:

        st.success(
            "🟢 LOW RISK OF HEART DISEASE"
        )


    st.info(
    "⚕️ This prediction is generated by a Machine Learning model "
    "and is intended for educational purposes only. "
    "It should not be considered a medical diagnosis."
)

# FOOTER

st.markdown(
    '<div class="footer">'
    '❤️ Heart Disease Prediction System &nbsp;|&nbsp; '
    'Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
