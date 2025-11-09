import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import h5py
import traceback
import json
import os
from pathlib import Path
from tensorflow.keras.models import load_model, model_from_json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

BASE = Path(__file__).parent

# Define file paths
LE_FP = BASE / "label_encoder_gender.pkl"
OHE_FP = BASE / "onehot_encoder_geo.pkl"
SCALER_FP = BASE / "scaler.pkl"
FEATURES_FP = BASE / "features_order.pkl"  # Optional file
MODEL_FP = BASE / "model.keras"

# --- load artifacts (fix missing encoder/model variables) ---
model = None
label_encoder_gender = None
onehot_encoder_geo = None
scaler = None
FEATURE_ORDER = None

# helper to safe-load pickles
def _safe_load_pickle(fp):
    try:
        with open(fp, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

# load label encoder
label_encoder_gender = _safe_load_pickle(LE_FP)
if label_encoder_gender is None:
    # fallback default so app doesn't crash; keep consistent shape for .classes_
    le = LabelEncoder()
    le.classes_ = np.array(["Male", "Female"])
    label_encoder_gender = le

# load onehot encoder
onehot_encoder_geo = _safe_load_pickle(OHE_FP)
if onehot_encoder_geo is None:
    # fallback: create a minimal OneHotEncoder with expected API
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    # fit on a default list so categories_ and get_feature_names_out work
    ohe.fit(np.array([["France"], ["Spain"], ["Germany"]]))
    onehot_encoder_geo = ohe

# load scaler
scaler = _safe_load_pickle(SCALER_FP)

# load feature order (optional)
FEATURE_ORDER = _safe_load_pickle(FEATURES_FP)

# load model (try Keras saved model or HDF5)
model_file_loaded = None
try:
    if MODEL_FP.exists():
        try:
            model = load_model(MODEL_FP)
            model_file_loaded = MODEL_FP.name
        except Exception:
            model = None
    else:
        # Try loading model.h5 as fallback
        h5_fp = BASE / "model.h5"
        if h5_fp.exists():
            try:
                model = load_model(h5_fp)
                model_file_loaded = h5_fp.name
            except Exception:
                model = None
        else:
            model = None
except Exception:
    model = None

# Sidebar status/help (provides placeholders and artifact status for the user)
st.sidebar.markdown("## Model & Artifacts")
st.sidebar.write(f"Model file: {model_file_loaded if model_file_loaded else 'Not found'}")
if model is not None:
    st.sidebar.success("Keras model ready")
else:
    st.sidebar.warning("Model not loaded — predictions disabled")

st.sidebar.write(f"Label encoder: {'loaded' if label_encoder_gender is not None else 'missing'}")
st.sidebar.write(f"OneHot encoder: {'loaded' if onehot_encoder_geo is not None else 'missing'}")
st.sidebar.write(f"Scaler: {'loaded' if scaler is not None else 'missing'}")

if st.sidebar.button("Re-run"):
    st.experimental_rerun()

st.set_page_config(page_title="Churn Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      /* page background */
      .stApp { background: linear-gradient(180deg,#f8fafc 0%, #ffffff 100%); font-family: "Inter", sans-serif; }
      /* card */
      .card { padding:20px; border-radius:12px; box-shadow: 0 10px 30px rgba(2,6,23,0.08); background: #ffffff; }
      /* header */
      .app-header { display:flex; gap:16px; align-items:center; margin-bottom:8px; }
      .app-title { font-size:20px; font-weight:700; color:#0f172a; }
      .app-sub { color:#6b7280; font-size:13px; margin-top:-4px; }
      /* small muted */
      .small-muted { color:#6b7280; font-size:13px; }
      /* inputs */
      .stSelectbox > div[role="button"] { border-radius:10px !important; }
      .stSlider > div[role="slider"] { accent-color:#ef4444; }
      /* metric card */
      .metric-card { padding:12px; border-radius:10px; background: linear-gradient(180deg,#ffffff,#fbfcfe); box-shadow:0 6px 18px rgba(15,23,42,0.04); }
    </style>
    """,
    unsafe_allow_html=True,
)

# helper: presets & randomize
import random
PRESETS = {
    "Default": {"Geography": "France", "Gender": "Male", "Age": 35, "Tenure": 3, "Balance": 1000.0,
                "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 5000.0},
    "High Risk": {"Geography": "Germany", "Gender": "Female", "Age": 60, "Tenure": 1, "Balance": 90000.0,
                  "NumOfProducts": 1, "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 2000.0},
    "Low Risk": {"Geography": "France", "Gender": "Male", "Age": 30, "Tenure": 7, "Balance": 2000.0,
                 "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 90000.0},
}

def apply_preset_to_session(p):
    for k, v in p.items():
        st.session_state.setdefault(f"inp_{k}", v)

def randomize_inputs():
    geo_list = list(onehot_encoder_geo.categories_[0]) if 'onehot_encoder_geo' in globals() else ["France","Spain","Germany"]
    gender_list = list(label_encoder_gender.classes_) if 'label_encoder_gender' in globals() else ["Male","Female"]
    s = {
        "Geography": random.choice(geo_list),
        "Gender": random.choice(gender_list),
        "Age": random.randint(18, 80),
        "Tenure": random.randint(0,10),
        "Balance": round(random.uniform(0, 100000), 2),
        "NumOfProducts": random.choice([1,2,3,4]),
        "HasCrCard": random.choice([0,1]),
        "IsActiveMember": random.choice([0,1]),
        "EstimatedSalary": round(random.uniform(1000, 150000), 2)
    }
    for k,v in s.items():
        st.session_state[f"inp_{k}"] = v

# ensure session state defaults exist
for k, v in PRESETS["Default"].items():
    st.session_state.setdefault(f"inp_{k}", v)

# Header + two columns layout
st.markdown("<div class='card'>", unsafe_allow_html=True)
col_header, col_space = st.columns([1, 3])
with col_header:
    st.markdown("<div class='app-header'>", unsafe_allow_html=True)
    st.image("https://static.streamlit.io/examples/dice.jpg", width=56)  # lightweight sample image; replace with project logo if available
    st.markdown("<div><div class='app-title'>Churn Predictor</div><div class='app-sub'>Interactive customer churn scoring</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_space:
    # preset buttons and randomize
    p1, p2, p3, p4 = st.columns([1,1,1,1])
    if p1.button("Default"):
        apply_preset_to_session(PRESETS["Default"])
        st.experimental_rerun()
    if p2.button("Low Risk"):
        apply_preset_to_session(PRESETS["Low Risk"])
        st.experimental_rerun()
    if p3.button("High Risk"):
        apply_preset_to_session(PRESETS["High Risk"])
        st.experimental_rerun()
    if p4.button("Randomize"):
        randomize_inputs()
        st.experimental_rerun()
st.markdown("</div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Customer Input")
    st.markdown("<div class='small-muted'>Provide customer attributes and press Predict</div>", unsafe_allow_html=True)

    # use a named form but bind initial values to session_state keys
    with st.form("input_form"):
        try:
            geo_options = list(onehot_encoder_geo.categories_[0])
        except Exception:
            geo_options = ["France", "Spain", "Germany"]
        try:
            gender_options = list(label_encoder_gender.classes_)
        except Exception:
            gender_options = ["Male", "Female"]

        geography = st.selectbox("Geography", geo_options, index=geo_options.index(st.session_state.get("inp_Geography")) if st.session_state.get("inp_Geography") in geo_options else 0, key="inp_Geography")
        gender = st.selectbox("Gender", gender_options, index=gender_options.index(st.session_state.get("inp_Gender")) if st.session_state.get("inp_Gender") in gender_options else 0, key="inp_Gender")
        age = st.slider("Age", 18, 100, value=int(st.session_state.get("inp_Age", 35)), key="inp_Age")
        tenure = st.slider("Tenure (years)", 0, 10, value=int(st.session_state.get("inp_Tenure", 3)), key="inp_Tenure")
        balance = st.number_input("Balance", min_value=0.0, value=float(st.session_state.get("inp_Balance", 1000.0)), format="%.2f", key="inp_Balance")
        num_of_products = st.selectbox("Number of Products", [1,2,3,4], index=[1,2,3,4].index(int(st.session_state.get("inp_NumOfProducts",1))), key="inp_NumOfProducts")
        has_cr_card = st.radio("Has Credit Card", [0,1], horizontal=True, index=int(st.session_state.get("inp_HasCrCard",1)), key="inp_HasCrCard")
        is_active_member = st.radio("Is Active Member", [0,1], horizontal=True, index=int(st.session_state.get("inp_IsActiveMember",1)), key="inp_IsActiveMember")
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=float(st.session_state.get("inp_EstimatedSalary",5000.0)), format="%.2f", key="inp_EstimatedSalary")

        submit = st.form_submit_button("Predict", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction")
    result_area = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

if submit:
    input_df = pd.DataFrame([{
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }])

    # encode gender using your label encoder
    try:
        input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])
    except Exception as e:
        st.sidebar.exception(e)
        st.error("Gender encoding failed.")
        st.stop()

    # one-hot geography using your encoder
    try:
        geo_enc = onehot_encoder_geo.transform(input_df[["Geography"]])
        # Handle both sparse matrix and dense array
        if hasattr(geo_enc, 'toarray'):
            geo_enc = geo_enc.toarray()
        geo_cols = onehot_encoder_geo.get_feature_names_out(["Geography"])
        geo_df = pd.DataFrame(geo_enc, columns=geo_cols, index=input_df.index)
    except Exception as e:
        st.sidebar.exception(e)
        st.error("Geography encoding failed.")
        st.stop()

    prepared_df = pd.concat([input_df.drop(columns=["Geography"]), geo_df], axis=1)

    # enforce saved feature order if available
    if FEATURE_ORDER is not None:
        missing = [c for c in FEATURE_ORDER if c not in prepared_df.columns]
        if missing:
            st.error(f"Missing features required by the model: {missing}")
            st.stop()
        prepared_df = prepared_df.reindex(columns=FEATURE_ORDER)

    # if scaler present, check feature count and apply; otherwise attempt prediction directly
    if scaler is not None:
        expected_n = None
        if hasattr(scaler, "n_features_in_"):
            expected_n = int(scaler.n_features_in_)
        elif hasattr(scaler, "mean_"):
            expected_n = scaler.mean_.shape[0]
        if expected_n is not None and prepared_df.shape[1] != expected_n:
            st.warning(f"Feature count mismatch: prepared={prepared_df.shape[1]} expected={expected_n}")
            # try recover with scaler.feature_names_in_
            if hasattr(scaler, "feature_names_in_"):
                feat_from_scaler = list(scaler.feature_names_in_)
                missing_cols = [c for c in feat_from_scaler if c not in prepared_df.columns]
                for c in missing_cols:
                    prepared_df[c] = 0.0
                prepared_df = prepared_df.reindex(columns=feat_from_scaler)
                st.info("Recovered columns from scaler.feature_names_in_; filled missing with zeros.")
            else:
                n_missing = expected_n - prepared_df.shape[1]
                if n_missing > 0:
                    for i in range(n_missing):
                        prepared_df[f"_placeholder_{i}"] = 0.0
                    st.info(f"Added {n_missing} placeholder columns with zeros.")
                else:
                    st.error("Prepared features exceed expected count; stop.")
                    st.stop()
        try:
            X_in = scaler.transform(prepared_df.astype(float))
        except Exception as e:
            st.sidebar.exception(e)
            st.error("Scaler.transform failed.")
            st.stop()
    else:
        # no scaler: force numeric and proceed (model may expect scaled input)
        X_in = prepared_df.astype(float).values

    try:
        raw_pred = model.predict(X_in)
        probability = float(np.asarray(raw_pred).reshape(-1)[0])
        probability = np.clip(probability, 0.0, 1.0)
    except Exception as e:
        st.sidebar.exception(e)
        st.error("Model prediction failed. See sidebar diagnostics.")
        st.stop()

    # display results
    with result_area.container():
        st.metric(label="Churn probability", value=f"{probability:.2%}")
        fig = px.pie(names=["Churn", "Stay"], values=[probability, 1.0 - probability], hole=0.6,
                     color_discrete_sequence=["#ef4444", "#10b981"])
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        if probability > 0.5:
            st.error("High churn risk — consider retention actions.")
        elif probability > 0.25:
            st.warning("Moderate churn risk — monitor and engage.")
        else:
            st.success("Low churn risk — customer likely to stay.")

    with st.expander("Prepared features (debug)"):
        st.write(prepared_df.T)
        st.caption(f"Model file used: {model_file_loaded if model_file_loaded else 'Not found'}")
