

import pandas as pd
import numpy as np
import pickle
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# Model Artifacts Loading
print("Loading model artifacts...")
try:
    with open('churn_model_artifacts.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    model = loaded_data['model']
    encoder = loaded_data['encoder']
    te_maps = loaded_data['te_maps']
    global_mean = loaded_data['global_mean']
    feature_cols = loaded_data['feature_cols']
    print("Model artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: 'churn_model_artifacts.pkl' not found. Please ensure it's in the same directory.")
    print("Using dummy model and target encoding maps for demonstration purposes.")

    model = LogisticRegression()

    te_maps = {
        'INCOME_LEVEL': {'Low': 0.1, 'Medium': 0.2, 'High': 0.3, 'Very High': 0.4},
        'EDUCATION': {'High School': 0.1, 'Graduate': 0.2, 'Post-Graduate': 0.3, 'Other': 0.15},
        'DEVICE_TYPE': {'Mobile-only': 0.1, 'Desktop-only': 0.2, 'Multi-device': 0.3}
    }
    global_mean = 0.2

    feature_cols = [
        'TOTAL_NUM_SESSIONS', 'GROSS_TOTAL_SESSION_LENGTH', 'ACTIVE_DAYS', 'ACTIVE_QUARTERS',
        'AVG_SESSIONS_PER_ACTIVE_QUARTER', 'AVG_SESSION_LENGTH_PER_ACTIVE_DAY', 'AGE', 'TECH_COMFORT_SCORE',
        'LOG_TOTAL_NUM_SESSIONS', 'LOG_GROSS_TOTAL_SESSION_LENGTH', 'LOG_ACTIVE_DAYS',
        'LOG_AVG_SESSIONS_PER_ACTIVE_QUARTER', 'LOG_AVG_SESSION_LENGTH_PER_ACTIVE_DAY',
        'SESSIONS_X_TECHCOMFORT', 'SESSION_LEN_X_TECHCOMFORT', 'ACTIVE_DAYS_X_QUARTERS',
        'SESSION_INTENSITY', 'ENGAGEMENT_SCORE', 'TE_INCOME_LEVEL', 'TE_EDUCATION', 'TE_DEVICE_TYPE'
    ]

    n_samples = 10 
    X_dummy = pd.DataFrame(np.random.rand(n_samples, len(feature_cols)), columns=feature_cols)
    y_dummy = np.random.randint(0, 2, n_samples) 
    model.fit(X_dummy, y_dummy) 

def predict(
    age: float,
    tech_comfort_score: float,
    total_num_sessions: float,
    gross_total_session_length: float,
    active_days: float,
    active_quarters: float,
    session_intensity: float,
    engagement_score: float,
    income_level: str,
    education: str,
    device_type: str
) -> str:
    """
    Predicts churn probability for a single customer based on input features.
    Applies feature engineering and encoding consistent with the trained model.
    """
    customer_data = pd.DataFrame([{
        'AGE': age,
        'TECH_COMFORT_SCORE': tech_comfort_score,
        'TOTAL_NUM_SESSIONS': total_num_sessions,
        'GROSS_TOTAL_SESSION_LENGTH': gross_total_session_length,
        'ACTIVE_DAYS': active_days,
        'ACTIVE_QUARTERS': active_quarters,
        'SESSION_INTENSITY': session_intensity,
        'ENGAGEMENT_SCORE': engagement_score,
        'INCOME_LEVEL': income_level,
        'EDUCATION': education,
        'DEVICE_TYPE': device_type
    }])

    processed_df = customer_data.copy()

    for col, mapping in te_maps.items():
        processed_df[f'TE_{col}'] = processed_df[col].map(mapping).fillna(global_mean)

    processed_df['LOG_TOTAL_NUM_SESSIONS'] = np.log1p(processed_df['TOTAL_NUM_SESSIONS'])
    processed_df['LOG_GROSS_TOTAL_SESSION_LENGTH'] = np.log1p(processed_df['GROSS_TOTAL_SESSION_LENGTH'])
    processed_df['LOG_ACTIVE_DAYS'] = np.log1p(processed_df['ACTIVE_DAYS'])

    processed_df['AVG_SESSIONS_PER_ACTIVE_QUARTER'] = processed_df.apply(
        lambda row: row['TOTAL_NUM_SESSIONS'] / row['ACTIVE_QUARTERS'] if row['ACTIVE_QUARTERS'] > 0 else 0, axis=1
    )
    processed_df['AVG_SESSION_LENGTH_PER_ACTIVE_DAY'] = processed_df.apply(
        lambda row: row['GROSS_TOTAL_SESSION_LENGTH'] / row['ACTIVE_DAYS'] if row['ACTIVE_DAYS'] > 0 else 0, axis=1
    )

    processed_df['LOG_AVG_SESSIONS_PER_ACTIVE_QUARTER'] = np.log1p(processed_df['AVG_SESSIONS_PER_ACTIVE_QUARTER'])
    processed_df['LOG_AVG_SESSION_LENGTH_PER_ACTIVE_DAY'] = np.log1p(processed_df['AVG_SESSION_LENGTH_PER_ACTIVE_DAY'])

    processed_df['SESSIONS_X_TECHCOMFORT'] = processed_df['TOTAL_NUM_SESSIONS'] * processed_df['TECH_COMFORT_SCORE']
    processed_df['SESSION_LEN_X_TECHCOMFORT'] = processed_df['GROSS_TOTAL_SESSION_LENGTH'] * processed_df['TECH_COMFORT_SCORE']
    processed_df['ACTIVE_DAYS_X_QUARTERS'] = processed_df['ACTIVE_DAYS'] * processed_df['ACTIVE_QUARTERS']

    final_features_df = pd.DataFrame(index=processed_df.index)
    for col in feature_cols:
        if col in processed_df.columns:
            final_features_df[col] = processed_df[col]
        else:
            final_features_df[col] = 0.0 

    renewal_probability = model.predict_proba(final_features_df[feature_cols])[:, 1][0]

    risk = 'Low' if renewal_probability >= 0.6 else 'Medium' if renewal_probability >= 0.4 else 'High'
    return f'Renewal Probability: {renewal_probability:.2f}  |  Churn Risk: {risk}'


# Streamlit Interface
st.set_page_config(layout="wide")
st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

with st.form("prediction_form"): 
    st.subheader("Customer Attributes")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", min_value=18, max_value=90, step=1, value=30)
        total_num_sessions = st.slider("Total Number of Sessions", min_value=0, max_value=1000, step=10, value=200)
        active_days = st.slider("Active Days", min_value=0, max_value=90, step=1, value=30)
        session_intensity = st.slider("Session Intensity", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
        income_level = st.radio("Income Level", ["Low", "Medium", "High", "Very High"], index=1) 

    with col2:
        tech_comfort_score = st.slider("Tech Comfort Score", min_value=1, max_value=10, step=1, value=5)
        gross_total_session_length = st.slider("Gross Total Session Length (minutes)", min_value=0, max_value=20000, step=100, value=5000)
        active_quarters = st.slider("Active Quarters", min_value=0, max_value=4, step=1, value=2)
        engagement_score = st.slider("Engagement Score", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
        education = st.radio("Education", ["High School", "Other", "Graduate", "Post-Graduate"], index=2) 

    with col3:
        device_type = st.radio("Device Type", ["Multi-device", "Mobile-only", "Desktop-only"], index=0) 

    st.markdown("--- ") 
    submitted = st.form_submit_button("Predict Renewal Probability")

if submitted:
    with st.spinner('Predicting...'):
        result = predict(
            age, tech_comfort_score, total_num_sessions,
            gross_total_session_length, active_days,
            active_quarters, session_intensity, engagement_score,
            income_level, education, device_type
        )
        st.success("Prediction Complete!")
        st.markdown(f"## {result}")
