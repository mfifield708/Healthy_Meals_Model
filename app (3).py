
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost # Make sure xgboost is imported for XGBClassifier
from sklearn.preprocessing import OneHotEncoder # Make sure OneHotEncoder is imported

# --- Load Model and Artifacts ---

# Assuming churn_model_artifacts.pkl is in the same directory as app.py
# Or, provide a full path if it's elsewhere.
try:
    with open('churn_model_artifacts.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    model = loaded_data['model']
    encoder = loaded_data['encoder']
    te_maps = loaded_data['te_maps']
    global_mean = loaded_data['global_mean']
    median_session_len = loaded_data['median_session_len']
    feature_cols = loaded_data['feature_cols']

    st.success("Model and artifacts loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'churn_model_artifacts.pkl' not found. Please ensure it's in the same directory as `app.py` or provide the correct path.")
    st.stop()
except KeyError as e:
    st.error(f"Error loading artifact: {e}. Check if all required keys are present in the pickle file.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during model loading: {e}")
    st.stop()

# --- Feature Engineering Functions (Recreating original preprocessing) ---

def apply_feature_engineering(df):
    # Ensure required columns are present in the input DataFrame
    required_for_engineering = [
        'TOTAL_NUM_SESSIONS', 'GROSS_TOTAL_SESSION_LENGTH', 'ACTIVE_DAYS',
        'ACTIVE_QUARTERS', 'AVG_SESSIONS_PER_ACTIVE_QUARTER',
        'AVG_SESSION_LENGTH_PER_ACTIVE_DAY', 'AGE', 'TECH_COMFORT_SCORE',
        'INCOME_LEVEL', 'EDUCATION', 'DEVICE_TYPE'
    ]
    for col in required_for_engineering:
        if col not in df.columns:
            st.warning(f"Missing column for feature engineering: {col}. Setting to 0 or 'Unknown'.")
            if col in ['INCOME_LEVEL', 'EDUCATION', 'DEVICE_TYPE']:
                df[col] = df[col].fillna('Unknown') # Handle missing categorical
            else:
                df[col] = df[col].fillna(0) # Handle missing numerical


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

    # Engineered features (assuming these are defined as combinations of others)
    # These are placeholders; adjust according to your actual feature engineering logic
    df['SESSION_INTENSITY'] = df['TOTAL_NUM_SESSIONS'] / (df['ACTIVE_DAYS'] + 1e-6) # Avoid division by zero
    df['ENGAGEMENT_SCORE'] = (df['TOTAL_NUM_SESSIONS'] * df['AVG_SESSION_LENGTH_PER_ACTIVE_DAY']) / (df['AGE'] + 1e-6) # Placeholder

    # Apply Target Encoding
    for col, te_map in te_maps.items():
        df[col] = df[col].map(te_map).fillna(global_mean) # Fill NaNs with global mean or a more appropriate value

    # Apply One-Hot Encoding
    # Assuming `encoder` was fitted on specific categorical columns not covered by te_maps.
    # Without knowing the original columns for OHE, this is a placeholder.
    # You might need to adjust `ohe_cols` based on your training data.
    # For this example, let's assume no additional OHE is needed beyond te_maps, or that
    # the `feature_cols` list will filter out non-existent OHE columns later.
    # If you had other categorical columns to OHE, you'd add them here:
    # ohe_cols = ['other_categorical_feature']
    # if not df[ohe_cols].empty:
    #     ohe_features = encoder.transform(df[ohe_cols])
    #     ohe_df = pd.DataFrame(ohe_features, columns=encoder.get_feature_names_out(ohe_cols))
    #     df = pd.concat([df.reset_index(drop=True), ohe_df], axis=1)

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

# --- Streamlit UI ---
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability.")

# Input fields for raw features
# Use st.sidebar for inputs to keep the main area clean
st.sidebar.header("Customer Data Input")

total_sessions = st.sidebar.number_input("Total Number of Sessions", min_value=0, value=50)
gross_session_length = st.sidebar.number_input("Gross Total Session Length (minutes)", min_value=0, value=300)
active_days = st.sidebar.number_input("Active Days", min_value=0, value=10)
active_quarters = st.sidebar.number_input("Active Quarters", min_value=0, value=2)
avg_sessions_per_quarter = st.sidebar.number_input("Avg Sessions per Active Quarter", min_value=0.0, value=25.0)
avg_session_len_per_day = st.sidebar.number_input("Avg Session Length per Active Day (minutes)", min_value=0.0, value=30.0)
age = st.sidebar.number_input("Age", min_value=18, value=35)
tech_comfort_score = st.sidebar.slider("Tech Comfort Score (1-10)", min_value=1, max_value=10, value=7)

income_level = st.sidebar.selectbox("Income Level", list(te_maps['INCOME_LEVEL'].keys()))
education = st.sidebar.selectbox("Education", list(te_maps['EDUCATION'].keys()))
device_type = st.sidebar.selectbox("Device Type", list(te_maps['DEVICE_TYPE'].keys()))

# Create a DataFrame from user inputs
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

if st.sidebar.button("Predict Churn"):
    # Preprocess the input data
    processed_input = preprocess_input(user_input_df)

    # Make prediction
    prediction_proba = model.predict_proba(processed_input)[:, 1][0] # Get probability of churn
    prediction_class = model.predict(processed_input)[0]

    st.subheader("Prediction Results:")
    st.write(f"Churn Probability: **{prediction_proba:.2f}**")

    if prediction_class == 1:
        st.error("This customer is predicted to **churn**.")
    else:
        st.success("This customer is predicted to **not churn**.")

    st.write("--- Debug Information ---")
    st.write("Processed Input Data:")
    st.dataframe(processed_input)
