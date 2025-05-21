import streamlit as st
import pandas as pd
import numpy as np
import joblib

# FEATURE COLUMNS
# Copy‐paste all columns from your notebook’s hot_encoding.columns (except 'log_salary'),
# in the exact order they appeared. For example:
feature_columns = [
    'Games', 'Minutes Played', 'Fields Goal', 'Fields Goal Attempted',
       '3-points Field Goal', '3-points Field Goal Attempted',
       '2-points Field Goal', '2-points Field Goal Attempted', 'Free Throws',
       'Free Throws Attempted', 'Offensive Rebounds', 'Defensive Rebounds',
       'Total Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers',
       'Personal Fouls', 'Points', 'Pos_C-PF', 'Pos_PF',
       'Pos_PF-C', 'Pos_PF-SF', 'Pos_PG', 'Pos_PG-SG', 'Pos_SF', 'Pos_SF-PF',
       'Pos_SF-SG', 'Pos_SG', 'Pos_SG-PG', 'Pos_SG-SF', 'Team_BOS', 'Team_BRK',
       'Team_CHI', 'Team_CHO', 'Team_CLE', 'Team_DAL', 'Team_DEN', 'Team_DET',
       'Team_GSW', 'Team_HOU', 'Team_IND', 'Team_LAC', 'Team_LAL', 'Team_MEM',
       'Team_MIA', 'Team_MIL', 'Team_MIN', 'Team_NOP', 'Team_NYK', 'Team_OKC',
       'Team_ORL', 'Team_PHI', 'Team_PHO', 'Team_POR', 'Team_SAC', 'Team_SAS',
       'Team_TOR', 'Team_TOT', 'Team_UTA', 'Team_WAS'
]

# CONTINUOUS COLUMNS
# List only the numeric (non‑binary, non‑target) columns that were scaled in training.
continuous_columns = [
    # Replace these with exactly what you used in the notebook:
    'Games', 'Minutes Played', 'Fields Goal', 'Fields Goal Attempted',
    '3-points Field Goal', '3-points Field Goal Attempted',
    '2-points Field Goal', '2-points Field Goal Attempted',
    'Free Throws', 'Free Throws Attempted', 'Offensive Rebounds',
    'Defensive Rebounds', 'Total Rebounds', 'Assists', 'Steals',
    'Blocks', 'Turnovers', 'Personal Fouls', 'Points'
]

# LOAD MODEL & SCALER
@st.cache_resource
def load_model_and_scaler():
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return xgb_model, scaler

xgb_model, scaler = load_model_and_scaler()

# APP LAYOUT
st.title("🏀 NBA Player Salary Predictor")
st.markdown(
    """
    Enter a player’s stats below to predict their **log‑salary**.
    The model was trained on historical NBA data (it outputs log(salary)).
    """
)

# CATEGORY DROPDOWNS
# Identify one‑hot columns by prefix:
POS_COLUMNS  = [c for c in feature_columns if c.startswith("Pos_")]
TEAM_COLUMNS = [c for c in feature_columns if c.startswith("Team_")]

selected_position = st.selectbox(
    "Position", options=[c.replace("Pos_", "") for c in POS_COLUMNS]
)
selected_team = st.selectbox(
    "Team", options=[c.replace("Team_", "") for c in TEAM_COLUMNS]
)

# CONTINUOUS INPUTS
input_data = {}
for col in continuous_columns:
    input_data[col] = st.number_input(label=col, value=0.0, step=1.0, format="%.2f")

# BUILD INPUT ROW 
user_dict = {}
for col in feature_columns:
    if col in continuous_columns:
        user_dict[col] = input_data[col]
    elif col in POS_COLUMNS:
        user_dict[col] = 1 if (col.replace("Pos_", "") == selected_position) else 0
    elif col in TEAM_COLUMNS:
        user_dict[col] = 1 if (col.replace("Team_", "") == selected_team) else 0
    else:
        user_dict[col] = 0  # any other one‑hot (if applicable)

input_df = pd.DataFrame([user_dict], columns=feature_columns)
st.write("### Raw inputs (before scaling)")
st.dataframe(input_df)

# PREPROCESS (SCALING)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[continuous_columns] = scaler.transform(df_copy[continuous_columns])
    return df_copy

processed_df = preprocess(input_df)
st.write("### After scaling continuous features")
st.dataframe(processed_df)

# PREDICTION
if st.button("Predict log‑salary"):
    model_input = processed_df[feature_columns]  # ensure correct column order
    log_salary_pred = xgb_model.predict(model_input)[0]
    st.success(f"Predicted log‑salary: {log_salary_pred:.3f}")
    salary_pred = np.exp(log_salary_pred)
    st.info(f"≈ Predicted salary: ${salary_pred:,.2f}")
