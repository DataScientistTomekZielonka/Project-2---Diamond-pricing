import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Loading constants
from src.constants import (
    CUT_MAPPING,
    COLOR_MAPPING,
    CLARITY_MAPPING
)

# Loading functions
from src.functions import (
    transform_diamond_features,
    scale_for_clusters,
    soft_voting_classifier,
    predict
)

# Set Streamlit Page Config
st.set_page_config(page_title = "Diamonds' Pricing", page_icon = ":gem:", layout = "wide")

# Large, centered title
st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>
        Gems' Whisperer  -  diamonds' pricing model 💎
    </h1>
    <br><br><br>
    """,
    unsafe_allow_html = True
)

# Load test data for use in the scrollable table
@st.cache_data
def load_data():
    with open("data/diamonds_test_data.json", "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

df_test_data = load_data()
df_test_data = df_test_data.drop(columns = ['price'], axis = 1)

# Load models and scalers
clusters_scaler = joblib.load('models/clusters/clusters_scaler.joblib')
lgbm_class_model = joblib.load('models/clusters/model_lgbm.joblib')
logreg_class_model = joblib.load('models/clusters/model_logreg.joblib')
rf_class_model = joblib.load('models/clusters/model_rf.joblib')
xgb_class_model = joblib.load('models/clusters/model_xgb.joblib')
models = [lgbm_class_model, logreg_class_model, rf_class_model, xgb_class_model]

# Store session state
if "selected_diamond" not in st.session_state:
    st.session_state["selected_diamond"] = None

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

if "input_record_display" not in st.session_state:
    st.session_state["input_record_display"] = None

input_record = {}

# Sidebar Form
with st.sidebar:
    st.title("Let's look at your gem...")
    st.image("graphics/snatch-avi.jpg", use_container_width=True)
    st.write("Input manually diamond's parameters.")

    # Pre-populating sidebar form if a diamond was selected from the table
    default_values = st.session_state["selected_diamond"] if st.session_state["selected_diamond"] else {}

    with st.form(key="sidebar_form", clear_on_submit=True):
        # min and max values based on EDA from step_2_eda.ipynb
        carats = st.slider(
            label = "How many carats does the diamond have?",
            min_value = 0.1,
            max_value = 5.0,
            value = 0.1,
            step = 0.01
        )

        depth = st.slider(
            label = "What is the diamond's depth percentage?",
            min_value = 40.0,
            max_value = 80.0,
            value = 40.0,
            step = 1.0
        )

        table = st.slider(
            label = "What is the size of the diamond's table?",
            min_value = 40,
            max_value = 95,
            value = 35,
            step = 1
        )
        
        length = st.slider(
            label = "What is the diamond's length (mm)?",
            min_value = 3.5,
            max_value = 11.0,
            value = 3.5,
            step = 0.01
        )
        
        width = st.slider(
            label = "What is the diamond's width (mm)?",
            min_value = 3.5,
            max_value = 60.0,
            value = 3.5,
            step = 0.01
        )
        
        depth_mm = st.slider(
            label="What is the diamond's depth (mm)?",
            min_value = 1.0,
            max_value = 32.0,
            value = 1.0,
            step = 0.01
        )
        
        cut = st.selectbox("Select the diamond's cut", options = CUT_MAPPING.keys())
        
        color = st.selectbox("Select the diamond's color", options = COLOR_MAPPING.keys())
        
        clarity = st.selectbox("Select the diamond's clarity", options = CLARITY_MAPPING.keys())

        # Collect input data
        input_record = {
            "carat": carats,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth,
            "table": table,
            "x": length,
            "y": width,
            "z": depth_mm
        }

        submit_button = st.form_submit_button(label="Submit")

# Process input when the form is submitted
if submit_button:
    st.sidebar.write("Form submitted successfully!")

    input_record_series = pd.Series(input_record)
    input_record_series_copy = input_record_series.copy()
    input_record_series = transform_diamond_features(input_record_series)

    input_record_df = pd.DataFrame([input_record_series])
    input_record_df_copy = pd.DataFrame([input_record_series_copy])  

    input_record_scaled_for_clusters = scale_for_clusters(clusters_scaler, input_record_df)
    input_record_scaled_for_clusters = pd.DataFrame(input_record_scaled_for_clusters, columns=input_record.keys())

    cluster = soft_voting_classifier(models, input_record_scaled_for_clusters)

    scaler = joblib.load(f'models/cluster {cluster}/scaler_{cluster}.joblib')
    model = joblib.load(f'models/cluster {cluster}/best_model/c{cluster}_best_model.joblib')

    prediction = int(np.round(predict(input_record_series, scaler, model), 0))

    input_record_display = input_record_df_copy.rename(columns={'depth': 'depth (%)', 'y': 'width (mm)', 'z': 'depth (mm)', 'x': 'length (mm)'})
    input_record_display.reset_index(drop=True, inplace=True)
    input_record_display = input_record_display.transpose()
    input_record_display.index.name = 'Feature'
    input_record_display.columns = ['Value']

    st.session_state["prediction"] = prediction
    st.session_state["input_record_display"] = input_record_display

# Create two columns for displaying forecast and table
col1, col2 = st.columns([1, 1])

# Left Column: Display Forecast
with col1:
    st.header("Estimated Price")
    st.image("graphics/snatch-frankie.jpg")

    # Check if there is a prediction in session state
    if st.session_state["prediction"] is not None:
        st.write(f'Estimated value : **${st.session_state["prediction"]:,}**')
        st.markdown(st.session_state["input_record_display"].to_markdown())
    else:
        # Display default text when no prediction is available
        st.write("Estimated value : **$ ---**")

        # Create a placeholder dataframe with headers
        empty_features_df = pd.DataFrame(columns=["Value"], index=["Carat", "Cut", "Color", "Clarity", "Depth (%)", "Table", "Length (mm)", "Width (mm)", "Depth (mm)"])
        empty_features_df.index.name = "Feature"  # Set index name
        empty_features_df["Value"] = "---"

        st.markdown(empty_features_df.to_markdown())

    # Add "Clear" button (resets col1 and resets selection in col2)
    if st.button("Clear"):
        st.session_state["prediction"] = None
        st.session_state["input_record_display"] = None

        # Reset col2 selection to row 0
        st.session_state["selected_index"] = 0
        st.session_state["selected_diamond"] = df_test_data.iloc[0].to_dict()

        st.rerun()


# Right Column: Scrollable Table for Selection
with col2:
    st.header("Select a Diamond from the Database")
    st.write("INFO : The below dataset was separated from initial data before training the model.")
    st.write("INFO : In real life scenario, this dataset can be replaced with the one for actual pricing.")

    # Ensure the selected index and diamond persist in session state
    if "selected_index" not in st.session_state:
        st.session_state["selected_index"] = 0  # Default row at start

    if "selected_diamond" not in st.session_state or not st.session_state["selected_diamond"]:
        st.session_state["selected_diamond"] = df_test_data.iloc[0].to_dict()  # Ensure it's always a valid dict

    # Dropdown to select a row (resets to 0 when "Clear" is clicked)
    selected_index = st.selectbox(
        "Choose a diamond row (row numbers: 0 - 4999)", 
        df_test_data.index, 
        index=st.session_state["selected_index"]  # Resets correctly
    )

    # Update session state only if a new row is selected
    if st.session_state["selected_index"] != selected_index:
        st.session_state["selected_index"] = selected_index
        st.session_state["selected_diamond"] = df_test_data.iloc[selected_index].to_dict()

    st.write("Selected Diamond Details:")
    
    # Ensure the JSON display always has a valid dictionary
    if isinstance(st.session_state["selected_diamond"], dict):
        st.json(st.session_state["selected_diamond"])
    else:
        st.write("No valid data available. Please select a valid row.")

    if st.button("Use This Diamond"):
        # Apply preprocessing and modeling
        selected_diamond_series = pd.Series(st.session_state["selected_diamond"])
        selected_diamond_series_copy = selected_diamond_series.copy()
        selected_diamond_series = transform_diamond_features(selected_diamond_series)

        selected_diamond_df = pd.DataFrame([selected_diamond_series])
        selected_diamond_scaled = scale_for_clusters(clusters_scaler, selected_diamond_df)

        selected_diamond_scaled_df = pd.DataFrame(
            selected_diamond_scaled, 
            columns=st.session_state["selected_diamond"].keys()
        )

        selected_cluster = soft_voting_classifier(models, selected_diamond_scaled_df)

        selected_scaler = joblib.load(f'models/cluster {selected_cluster}/scaler_{selected_cluster}.joblib')
        selected_model = joblib.load(f'models/cluster {selected_cluster}/best_model/c{selected_cluster}_best_model.joblib')

        selected_prediction = int(np.round(predict(selected_diamond_series, selected_scaler, selected_model), 0))

        selected_diamond_display = pd.DataFrame([selected_diamond_series_copy]).transpose()
        selected_diamond_display.index.name = "Feature"
        selected_diamond_display.columns = ['Value']

        st.session_state["prediction"] = selected_prediction
        st.session_state["input_record_display"] = selected_diamond_display

        st.rerun()
