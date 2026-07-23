
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import xgboost

# --- Load Model and Artifacts ---
# Assuming churn_model_artifacts.pkl is in the same directory as app.py
try:
    with open('churn_model_artifacts.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    model = loaded_data['model']
    te_maps = loaded_data['te_maps']
    global_mean = loaded_data['global_mean']
    median_session_len = loaded_data['median_session_len']
    feature_cols = loaded_data['feature_cols']
    print("Model and artifacts loaded successfully!")
except FileNotFoundError:
    print("Error: 'churn_model_artifacts.pkl' not found. Please ensure it's in the same directory.")
    exit()
except KeyError as e:
    print(f"Error loading artifact: {e}. Check if all required keys are present in the pickle file.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    exit()

# --- Feature Engineering Functions ---
def apply_feature_engineering(df):
    required_for_engineering = [
        'TOTAL_NUM_SESSIONS', 'GROSS_TOTAL_SESSION_LENGTH', 'ACTIVE_DAYS',
        'ACTIVE_QUARTERS', 'AVG_SESSIONS_PER_ACTIVE_QUARTER',
        'AVG_SESSION_LENGTH_PER_ACTIVE_DAY', 'AGE', 'TECH_COMFORT_SCORE',
        'INCOME_LEVEL', 'EDUCATION', 'DEVICE_TYPE'
    ]
    for col in required_for_engineering:
        if col not in df.columns:
            print(f"Missing column for feature engineering: {col}. Setting to 0 or 'Unknown'.")
            if col in ['INCOME_LEVEL', 'EDUCATION', 'DEVICE_TYPE']:
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(0)

    # Log transformations
    df['LOG_TOTAL_NUM_SESSIONS'] = np.log1p(df['TOTAL_NUM_SESSIONS'])
    df['LOG_GROSS_TOTAL_SESSION_LENGTH'] = np.log1p(df['GROSS_TOTAL_SESSION_LENGTH'])
    df['LOG_ACTIVE_DAYS'] = np.log1p(df['ACTIVE_DAYS'])
    df['LOG_AVG_SESSIONS_PER_ACTIVE_QUARTER'] = np.log1p(df['AVG_SESSIONS_PER_ACTIVE_QUARTER'])
    df['LOG_AVG_SESSION_LENGTH_PER_ACTIVE_DAY'] = np.log1p(df['AVG_SESSION_LENGTH_PER_ACTIVE_DAY'])

    # Interaction features
    df['SESSIONS_X_TECHCOMFORT'] = df['TOTAL_NUM_SESSIONS'] * df['TECH_COMFORT_SCORE']
    df['SESSION_LEN_X_TECHCOMFORT'] = df['GROSS_TOTAL_SESSION_LENGTH'] * df['TECH_COMFORT_SCORE']
    df['ACTIVE_DAYS_X_QUARTERS'] = df['ACTIVE_DAYS'] * df['ACTIVE_QUARTERS']

    # Engineered features
    df['SESSION_INTENSITY'] = df['TOTAL_NUM_SESSIONS'] / (df['ACTIVE_DAYS'] + 1e-6)
    df['ENGAGEMENT_SCORE'] = (df['TOTAL_NUM_SESSIONS'] * df['AVG_SESSION_LENGTH_PER_ACTIVE_DAY']) / (df['AGE'] + 1e-6)

    # Apply Target Encoding
    for col, te_map in te_maps.items():
        df[col] = df[col].map(te_map).fillna(global_mean) # Fill NaNs with global mean

    return df

def preprocess_input(user_input_df):
    processed_df = apply_feature_engineering(user_input_df.copy())

    # Select and reorder columns to match the training data's feature_cols
    final_df = pd.DataFrame(columns=feature_cols) # Create a DataFrame with the correct columns
    for col in feature_cols:
        if col in processed_df.columns:
            final_df[col] = processed_df[col]
        else:
            final_df[col] = 0 # Fill with a default value (e.g., 0) if a feature is missing

    return final_df

# --- Prediction Function for Gradio ---
def predict_churn(
    total_sessions,
    gross_session_length,
    active_days,
    active_quarters,
    avg_sessions_per_quarter,
    avg_session_len_per_day,
    age,
    education,
    income_level,
    device_type,
    tech_comfort_score
):
    user_input_df = pd.DataFrame({
        'TOTAL_NUM_SESSIONS': [total_sessions],
        'GROSS_TOTAL_SESSION_LENGTH': [gross_session_length],
        'ACTIVE_DAYS': [active_days],
        'ACTIVE_QUARTERS': [active_quarters],
        'AVG_SESSIONS_PER_ACTIVE_QUARTER': [avg_sessions_per_quarter],
        'AVG_SESSION_LENGTH_PER_ACTIVE_DAY': [avg_session_len_per_day],
        'AGE': [age],
        'TECH_COMFORT_SCORE': [tech_comfort_score],
        'INCOME_LEVEL': [income_level],
        'EDUCATION': [education],
        'DEVICE_TYPE': [device_type]
    })

    processed_input = preprocess_input(user_input_df)
    prediction_proba = model.predict_proba(processed_input)[:, 1][0]
    prediction_class = model.predict(processed_input)[0]

    return {"churn": prediction_proba, "not churn": 1 - prediction_proba}

# --- Gradio Interface ---
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Total Sessions (2022)", value=50),
        gr.Number(label="Total Session Length (minutes)", value=300),
        gr.Number(label="Active Days", value=10),
        gr.Slider(0, 4, step=1, label="Active Quarters", value=2),
        gr.Number(label="Avg Sessions per Active Quarter", value=25.0),
        gr.Number(label="Avg Session Length per Active Day", value=30.0),
        gr.Slider(18, 90, step=1, label="Age", value=35),
        gr.Dropdown(list(te_maps['EDUCATION'].keys()), label="Education", value='Graduate'),
        gr.Dropdown(list(te_maps['INCOME_LEVEL'].keys()), label="Income Level", value='High'),
        gr.Dropdown(list(te_maps['DEVICE_TYPE'].keys()), label="Device Type", value='Desktop-only'),
        gr.Slider(1, 10, step=1, label="Tech Comfort Score", value=7)
    ],
    outputs=gr.Label(num_top_classes=2),
    title="Healthy Meals Churn Predictor",
    description="Enter subscriber features to predict renewal probability."
)

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
