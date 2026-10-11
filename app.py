from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import urllib.request

MODEL_PATH = Path("models/flight_delay_model.joblib")
METRICS_PATH = Path("models/metrics.json")

st.set_page_config(page_title="Flight Delay Risk Predictor", page_icon="✈️", layout="wide")
st.title("✈️ Flight Delay Prediction and Disruption Risk Classification")
st.write("Enter the available flight details to estimate whether the flight may be delayed by more than 15 minutes.")
st.caption("This is an academic prototype. Predictions are estimates, not official airline information.")

if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_URL = "https://github.com/satwikhiremath/flight-delay-prediction/releases/download/v1.0.0/flight_delay_model.joblib"
    try:
        with st.spinner("Downloading the trained model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        st.error(f"Could not download the trained model: {e}")
        st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
numeric_columns = set(bundle["numeric_columns"])
defaults = bundle["feature_defaults"]

if METRICS_PATH.exists():
    metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.info(f"Loaded model: {metrics_data.get('best_model', bundle.get('best_model_name', 'Trained model'))}")

st.subheader("Flight details")
st.write("Fill in the fields you know. Leave the default values if you are unsure.")
values = {}
with st.form("prediction_form"):
    cols = st.columns(2)
    for i, name in enumerate(feature_columns):
        with cols[i % 2]:
            default = defaults.get(name, "Unknown")
            label = str(name).replace("_", " ").title()
            if name in numeric_columns:
                try:
                    values[name] = st.number_input(label, value=float(default), key=name)
                except Exception:
                    values[name] = st.number_input(label, value=0.0, key=name)
            else:
                values[name] = st.text_input(label, value=str(default), key=name)
    submitted = st.form_submit_button("Predict Delay Risk")

if submitted:
    row = pd.DataFrame([values], columns=feature_columns)
    try:
        prediction = int(pipeline.predict(row)[0])
        probability = float(pipeline.predict_proba(row)[0][1])
        if prediction == 1:
            st.error("Prediction: HIGHER DELAY RISK")
        else:
            st.success("Prediction: LOWER DELAY RISK")
        st.metric("Estimated probability of delay", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("Risk threshold is determined by the trained classifier. A result of 1 means delay risk; 0 means lower predicted risk.")
    except Exception as e:
        st.error(f"Could not create a prediction from these values: {e}")

if METRICS_PATH.exists():
    st.divider()
    st.subheader("Model evaluation")
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.write(f"Target column: `{data.get('target_column', 'unknown')}`")
    for model_name, m in data.get("results", {}).items():
        st.markdown(f"**{model_name}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']:.3f}")
        c2.metric("Precision", f"{m['precision']:.3f}")
        c3.metric("Recall", f"{m['recall']:.3f}")
        c4.metric("F1-score", f"{m['f1']:.3f}")
        c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    imp_path = Path("models/feature_importance.csv")
    if imp_path.exists():
        st.subheader("Top feature importance")
        st.dataframe(pd.read_csv(imp_path), use_container_width=True)
