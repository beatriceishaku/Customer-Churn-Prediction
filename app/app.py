import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

# Add this near the top of app.py, after imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # goes up from app/ to project root

@st.cache_resource
def load_model():
    with open(os.path.join(ROOT_DIR, 'models', 'xgb_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(ROOT_DIR, 'models', 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(ROOT_DIR, 'models', 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

model, scaler, feature_cols = load_model()

st.title("Customer Churn Predictor")
st.caption("Enter customer details to predict the probability they will cancel their subscription.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account info")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract type",
        ["Month-to-month", "One year", "Two year"])
    paperless = st.checkbox("Paperless billing", value=True)
    payment = st.selectbox("Payment method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

with col2:
    st.subheader("Services")
    internet = st.selectbox("Internet service",
        ["Fiber optic", "DSL", "No"])
    phone = st.checkbox("Phone service", value=True)
    multiple_lines = st.selectbox("Multiple lines",
        ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online security",
        ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online backup",
        ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Billing")
    monthly_charges = st.slider("Monthly charges ($)", 18, 120, 65)
    senior = st.checkbox("Senior citizen", value=False)
    partner = st.checkbox("Has partner", value=False)
    dependents = st.checkbox("Has dependents", value=False)
    streaming_tv = st.selectbox("Streaming TV",
        ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies",
        ["No", "Yes", "No internet service"])

total_charges = monthly_charges * tenure

st.markdown("---")

if st.button("Predict churn risk", type="primary"):

    input_dict = {col: 0 for col in feature_cols}

    num_fields = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'charges_per_month': monthly_charges,
        'is_new_customer': int(tenure < 6),
        'is_long_term': int(tenure > 36),
        'SeniorCitizen': int(senior),
        'has_multiple_services': (
            int(phone) +
            int(internet != 'No') +
            int(streaming_tv == 'Yes') +
            int(streaming_movies == 'Yes')
        ),
        'high_value': int(monthly_charges > 64),
    }
    for k, v in num_fields.items():
        if k in input_dict:
            input_dict[k] = v

    flag_fields = {
        'Partner_Yes': int(partner),
        'Dependents_Yes': int(dependents),
        'PhoneService_Yes': int(phone),
        'PaperlessBilling_Yes': int(paperless),
        f'MultipleLines_{multiple_lines}': 1,
        f'InternetService_{internet}': 1,
        f'OnlineSecurity_{online_security}': 1,
        f'OnlineBackup_{online_backup}': 1,
        f'StreamingTV_{streaming_tv}': 1,
        f'StreamingMovies_{streaming_movies}': 1,
        f'Contract_{contract}': 1,
        f'PaymentMethod_{payment}': 1,
    }
    for k, v in flag_fields.items():
        if k in input_dict:
            input_dict[k] = v

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_cols)

    prob = model.predict_proba(input_scaled_df)[0][1]
    pred = int(prob > 0.5)

    r1, r2, r3 = st.columns(3)

    with r1:
        color = "🔴" if prob > 0.6 else "🟡" if prob > 0.35 else "🟢"
        st.metric("Churn probability", f"{prob:.1%}")
        if prob > 0.6:
            st.error("High risk — this customer is likely to churn.")
        elif prob > 0.35:
            st.warning("Medium risk — monitor this customer.")
        else:
            st.success("Low risk — this customer is likely to stay.")

    with r2:
        risk_label = "High" if prob > 0.6 else "Medium" if prob > 0.35 else "Low"
        fig, ax = plt.subplots(figsize=(3, 3))
        colors_pie = ['#E24B4A', '#888780'] if pred else ['#378ADD', '#888780']
        ax.pie([prob, 1 - prob],
               labels=["Churn", "Stay"],
               colors=colors_pie,
               autopct='%1.1f%%',
               startangle=90)
        ax.set_title(f"Risk: {risk_label}")
        st.pyplot(fig)
        plt.close()

    with r3:
        st.markdown("**Key factors for this prediction:**")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(input_scaled_df)
        shap_series = pd.Series(sv[0], index=feature_cols)
        top = shap_series.abs().nlargest(5).index
        for feat in top:
            val = shap_series[feat]
            arrow = "▲ increases" if val > 0 else "▼ decreases"
            color_txt = "🔴" if val > 0 else "🟢"
            st.markdown(f"{color_txt} `{feat}` — {arrow} churn risk")

st.markdown("---")
st.caption("Model: XGBoost | Dataset: Telco Customer Churn (Kaggle) | Built for portfolio")