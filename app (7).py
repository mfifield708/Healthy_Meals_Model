import pandas as pd
import numpy as np
import pickle
import gradio as gr
from sklearn.linear_model import LogisticRegression # Needed for dummy model in fallback
from sklearn.preprocessing import OneHotEncoder # Needed for dummy encoder in fallback

# --- Model Artifacts Loading ---
print("Loading model artifacts...")
try:
    with open('churn_model_artifacts.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    model = loaded_data['model']
    encoder = loaded_data['encoder'] # OneHotEncoder
    te_maps = loaded_data['te_maps']
    global_mean = loaded_data['global_mean']
    feature_cols = loaded_data['feature_cols']
    print("Model artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: 'churn_model_artifacts.pkl' not found. Please ensure it's in the same directory.")
    print("Using dummy model and target encoding maps/encoder for demonstration purposes.")
    # Fallback for demonstration if file is missing
    model = LogisticRegression()
    encoder = OneHotEncoder(handle_unknown='ignore') # Dummy encoder
    te_maps = {
        'INCOME_LEVEL': {'Low': 0.1, 'Medium': 0.2, 'High': 0.3, 'Very High': 0.4},
        'EDUCATION': {'High School': 0.1, 'Graduate': 0.2, 'Post-Graduate': 0.3, 'Other': 0.15},
        'DEVICE_TYPE': {'Mobile-only': 0.1, 'Desktop-only': 0.2, 'Multi-device': 0.3}
    }
    global_mean = 0.2
    feature_cols = ['AGE', 'TECH_COMFORT_SCORE', 'INCOME_LEVEL_Medium', 'INCOME_LEVEL_High', 'INCOME_LEVEL_Very High','EDUCATION_Other', 'EDUCATION_Graduate', 'EDUCATION_Post-Graduate','DEVICE_TYPE_Mobile-only', 'DEVICE_TYPE_Multi-device'] # Dummy feature_cols
    # Fit dummy encoder on sample data to avoid errors if fallback is used
    sample_data = pd.DataFrame({
        'INCOME_LEVEL': ['Low', 'Medium', 'High', 'Very High'],
        'EDUCATION': ['High School', 'Other', 'Graduate', 'Post-Graduate'],
        'DEVICE_TYPE': ['Mobile-only', 'Desktop-only', 'Multi-device', 'Mobile-only']
    })
    encoder.fit(sample_data)

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
    # Create a DataFrame for the single customer with raw inputs
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

    # Apply Target Encoding
    for col, mapping in te_maps.items():
        processed_df[f'TE_{col}'] = processed_df[col].map(mapping).fillna(global_mean)

    # Engineered features (log transforms)
    processed_df['LOG_TOTAL_NUM_SESSIONS'] = np.log1p(processed_df['TOTAL_NUM_SESSIONS'])
    processed_df['LOG_GROSS_TOTAL_SESSION_LENGTH'] = np.log1p(processed_df['GROSS_TOTAL_SESSION_LENGTH'])
    processed_df['LOG_ACTIVE_DAYS'] = np.log1p(processed_df['ACTIVE_DAYS'])

    # Ratio features, handle division by zero
    processed_df['AVG_SESSIONS_PER_ACTIVE_QUARTER'] = processed_df.apply(
        lambda row: row['TOTAL_NUM_SESSIONS'] / row['ACTIVE_QUARTERS'] if row['ACTIVE_QUARTERS'] > 0 else 0, axis=1
    )
    processed_df['AVG_SESSION_LENGTH_PER_ACTIVE_DAY'] = processed_df.apply(
        lambda row: row['GROSS_TOTAL_SESSION_LENGTH'] / row['ACTIVE_DAYS'] if row['ACTIVE_DAYS'] > 0 else 0, axis=1
    )

    processed_df['LOG_AVG_SESSIONS_PER_ACTIVE_QUARTER'] = np.log1p(processed_df['AVG_SESSIONS_PER_ACTIVE_QUARTER'])
    processed_df['LOG_AVG_SESSION_LENGTH_PER_ACTIVE_DAY'] = np.log1p(processed_df['AVG_SESSION_LENGTH_PER_ACTIVE_DAY'])

    # Interaction features
    processed_df['SESSIONS_X_TECHCOMFORT'] = processed_df['TOTAL_NUM_SESSIONS'] * processed_df['TECH_COMFORT_SCORE']
    processed_df['SESSION_LEN_X_TECHCOMFORT'] = processed_df['GROSS_TOTAL_SESSION_LENGTH'] * processed_df['TECH_COMFORT_SCORE']
    processed_df['ACTIVE_DAYS_X_QUARTERS'] = processed_df['ACTIVE_DAYS'] * processed_df['ACTIVE_QUARTERS']

    # Prepare the final feature DataFrame for prediction, ensuring correct columns and order
    final_features_df = pd.DataFrame(index=processed_df.index)
    for col in feature_cols:
        if col in processed_df.columns:
            final_features_df[col] = processed_df[col]
        else:
            final_features_df[col] = 0.0

    # Predict: column 1 = P(renewed)
    renewal_probability = model.predict_proba(final_features_df[feature_cols])[:, 1][0]

    risk = "Low" if renewal_probability >= 0.6 else "Medium" if renewal_probability >= 0.4 else "High"
    return f"Renewal Probability: {renewal_probability:.2f}  |  Churn Risk: {risk}"


# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(minimum=18, maximum=90, step=1, value=30, label="Age"),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Tech Comfort Score"),
        gr.Slider(minimum=0, maximum=1000, step=10, value=200, label="Total Number of Sessions"),
        gr.Slider(minimum=0, maximum=20000, step=100, value=5000, label="Gross Total Session Length (minutes)"),
        gr.Slider(minimum=0, maximum=90, step=1, value=30, label="Active Days"),
        gr.Slider(minimum=0, maximum=4, step=1, value=2, label="Active Quarters"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Session Intensity"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Engagement Score"),
        gr.Radio(["Low", "Medium", "High", "Very High"] , label="Income Level"),
        gr.Radio(["High School", "Other", "Graduate", "Post-Graduate"] , label="Education"),
        gr.Radio(["Multi-device", "Mobile-only", "Desktop-only"] , label="Device Type"),
    ],
    outputs="text" ,
    title="Customer Renewal Probability Predictor" ,
    description="Enter customer attributes to predict the likelihood of subscription renewal."
)

if __name__ == '__main__':
    iface.launch()