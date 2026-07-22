
import gradio as gr
import numpy as np
import pandas as pd
import pickle

# --- Load models and artifacts ---
# Assuming these files are in the same directory as app.py
try:
    with open('xgb_model.pkl', 'rb') as f:
        best_xgb_model = pickle.load(f)
    with open('lgbm_model.pkl', 'rb') as f:
        best_lgbm_model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('target_encoding_maps.pkl', 'rb') as f:
        target_encoding_maps = pickle.load(f)
    print("Models and artifacts loaded successfully within app.py.")
except FileNotFoundError as e:
    print(f"Error loading model artifact: {e}. Please ensure all .pkl files are in the same directory as app.py.")
    # Fallback to dummy models for local testing if files are not present
    class DummyModel:
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                num_samples = X.shape[0]
            else:
                num_samples = len(X) if isinstance(X, list) else 1
            return np.array([[0.25, 0.75]] * num_samples) # Default to 75% churn

    class DummyEncoder:
        def transform(self, X):
            return X

    best_xgb_model = DummyModel()
    best_lgbm_model = DummyModel()
    encoder = DummyEncoder()
    target_encoding_maps = {
        'education': {'High School': 0.1, 'Bachelors': 0.2, 'Masters': 0.3, 'PhD': 0.4},
        'income_level': {'Low': 0.1, 'Medium': 0.2, 'High': 0.3},
        'device_type': {'Mobile': 0.1, 'Desktop': 0.2, 'Tablet': 0.3}
    }
    print("Using dummy models due to missing artifact files.")
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")

# --- Feature Engineering and Prediction Function ---
def predict_churn(total_sessions, total_session_length, active_days, 
                  active_quarters, avg_sessions_per_quarter, 
                  avg_session_length_per_day, age, education, 
                  income_level, device_type, tech_comfort_score):
    
    # Build feature vector (apply same transforms as training)
    features = {}
    features['TOTAL_NUM_SESSIONS'] = total_sessions
    features['GROSS_TOTAL_SESSION_LENGTH'] = total_session_length
    features['ACTIVE_DAYS'] = active_days
    features['ACTIVE_QUARTERS'] = active_quarters
    features['AVG_SESSIONS_PER_ACTIVE_QUARTER'] = avg_sessions_per_quarter
    features['AVG_SESSION_LENGTH_PER_ACTIVE_DAY'] = avg_session_length_per_day
    features['AGE'] = age
    features['TECH_COMFORT_SCORE'] = tech_comfort_score

    # Log transforms
    for col in ['TOTAL_NUM_SESSIONS', 'GROSS_TOTAL_SESSION_LENGTH', 'ACTIVE_DAYS',
                'AVG_SESSIONS_PER_ACTIVE_QUARTER', 'AVG_SESSION_LENGTH_PER_ACTIVE_DAY']:
        features[f'LOG_{col}'] = np.log1p(features[col])

    # Interaction features
    features['SESSIONS_X_TECHCOMFORT'] = total_sessions * tech_comfort_score
    features['SESSION_LEN_X_TECHCOMFORT'] = avg_session_length_per_day * tech_comfort_score
    features['ACTIVE_DAYS_X_QUARTERS'] = active_days * active_quarters
    features['SESSION_INTENSITY'] = total_sessions / (active_days + 1)
    features['ENGAGEMENT_SCORE'] = (total_sessions * total_session_length) / (active_days + 1)
    features['AGE_X_TECHCOMFORT'] = age * tech_comfort_score
    features['IS_INACTIVE'] = int(active_days == 0)
    features['IS_HIGH_ENGAGEMENT'] = int(avg_session_length_per_day > 45) # use training median

    # Handle categorical features using target_encoding_maps
    features['EDUCATION_ENCODED'] = target_encoding_maps['education'].get(education, 0.0) # Default to 0.0 if not found
    features['INCOME_LEVEL_ENCODED'] = target_encoding_maps['income_level'].get(income_level, 0.0)
    features['DEVICE_TYPE_ENCODED'] = target_encoding_maps['device_type'].get(device_type, 0.0)

    # Convert features to a DataFrame for model prediction
    feature_df = pd.DataFrame([features])

    # Align columns with training data using the encoder
    # Assuming the encoder's transform method handles missing columns by adding them as 0 or similar
    processed_features = encoder.transform(feature_df)

    # Predict probabilities using both models (e.g., ensemble by averaging)
    # Make sure to get the probability of the positive class (churned)
    xgb_prob = best_xgb_model.predict_proba(processed_features)[:, 1]
    lgbm_prob = best_lgbm_model.predict_proba(processed_features)[:, 1]

    # Simple averaging ensemble
    avg_prob = (xgb_prob + lgbm_prob) / 2
    prob = avg_prob[0] # Get the single prediction probability
    
    return {
        "Renewed (stays)": float(1 - prob),
        "Churned (leaves)": float(prob)
    }

# Gradio interface
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Total Sessions (2022)"),
        gr.Number(label="Total Session Length (minutes)"),
        gr.Number(label="Active Days"),
        gr.Slider(0, 4, step=1, label="Active Quarters"),
        gr.Number(label="Avg Sessions per Active Quarter"),
        gr.Number(label="Avg Session Length per Active Day"),
        gr.Slider(18, 90, step=1, label="Age"),
        gr.Dropdown(["High School", "Bachelors", "Masters", "PhD"], label="Education"),
        gr.Dropdown(["Low", "Medium", "High"], label="Income Level"),
        gr.Dropdown(["Mobile", "Desktop", "Tablet"], label="Device Type"),
        gr.Slider(1, 10, step=1, label="Tech Comfort Score"),
    ],
    outputs=gr.Label(num_top_classes=2),
    title="Healthy Meals Churn Predictor",
    description="Enter subscriber features to predict renewal probability."
)

demo.launch(share=False) # share=False for Hugging Face Spaces deployment
