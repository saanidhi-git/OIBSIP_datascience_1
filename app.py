import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Importing  custom functions from the local file
from iris_classifier import get_trained_model, get_data_frame 


st.set_page_config(
    page_title="Iris Species Classifier", 
    layout="wide", 
    initial_sidebar_state="auto"
)


APP_BACKGROUND_COLOR = "#f0f2f6"


st.markdown(f"""
<style>
/* --- Remove the white overlay from all Streamlit containers --- */
.stApp {{
    background: none !important;
}}

[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stMainBlockContainer"],
.block-container {{
    background-color: {APP_BACKGROUND_COLOR} !important;
    color: black !important;
}}

[data-testid="stHeader"],
[data-testid="stToolbar"] {{
    background: transparent !important;
    color: black !important;
}}

section.main > div:first-child {{
    background: none !important;
}}

h1 {{
    color: #4B0082;
    text-align: center;
}}

h3 {{
    color: #6A5ACD;
}}

.prediction-box {{
    padding: 30px;
    border-radius: 15px;
    background-color: #e6ffe6; 
    border-left: 8px solid #00AA00; 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    text-align: center;
    margin-top: 20px;
}}
.prediction-text {{
    font-size: 28px;
    font-weight: bold;
    color: #008000; 
}}

footer {{visibility: hidden;}}

div[data-testid="column"] {{
    padding-left: 5px !important;
    padding-right: 5px !important;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource 
def load_model_and_data():
    """Retrieves the trained model and initial DataFrame."""
    model, target_names, feature_columns = get_trained_model()
    df = get_data_frame()
    return model, target_names, feature_columns, df

model, target_names, feature_columns, df = load_model_and_data()

if model is None or df is None:
    st.error("🚨 Error: Could not load model or data. Check 'iris_classifier.py' and ensure 'Iris.csv' is present.")
    st.stop() 



st.sidebar.header("Project Overview")
st.sidebar.info("This application uses a K-Nearest Neighbors (KNN) model trained on the classic Iris flower dataset.")
st.sidebar.metric(label="Model Accuracy (on Test Data)", value="100.00%", delta="Perfect Score", delta_color="normal")
st.sidebar.markdown('***')
st.sidebar.text("Model Details:")
st.sidebar.text(f"Algorithm: KNN (k=5)")
st.sidebar.text(f"Data Source: Iris.csv")
st.sidebar.text(f"Logic File: iris_classifier.py")



st.title('Iris Flower Species Classifier 🌸')
st.markdown('### 1. Input Measurements for Prediction')
st.markdown("Use the controls below to set the Sepal and Petal dimensions (in cm).")

# Define the feature details
feature_details = {
    'SepalLengthCm': {'label': 'Sepal Length (cm)', 'min': 4.0, 'max': 8.0, 'default': 5.4, 'help': 'Length of the sepal petal (outer leaf).'},
    'SepalWidthCm': {'label': 'Sepal Width (cm)', 'min': 2.0, 'max': 4.5, 'default': 3.4, 'help': 'Width of the sepal petal (outer leaf).'},
    'PetalLengthCm': {'label': 'Petal Length (cm)', 'min': 1.0, 'max': 7.0, 'default': 1.3, 'help': 'Length of the petal (inner leaf).'},
    'PetalWidthCm': {'label': 'Petal Width (cm)', 'min': 0.1, 'max': 2.5, 'default': 0.2, 'help': 'Width of the petal (inner leaf).'}
}

input_values = {}
input_cols = st.columns(len(feature_details))

for i, col_name in enumerate(feature_details.keys()):
    details = feature_details[col_name]
    with input_cols[i]:
        st.markdown(f"**{details['label']}**")
        typed_value = st.number_input(
            label='Enter value:',
            min_value=details['min'], 
            max_value=details['max'], 
            value=details['default'], 
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
            key=f'{col_name}_text_input',
            help=details['help']
        )
        slider_value = st.slider(
            label='Adjust Slider:',
            min_value=details['min'], 
            max_value=details['max'], 
            value=typed_value, 
            step=0.1,
            label_visibility="collapsed",
            key=f'{col_name}_slider'
        )
        input_values[col_name] = slider_value

st.markdown('---')

#  Prediction Logic and Output ---
if st.button('🚀 Run Classification', type="primary", use_container_width=True):
    try:
        new_flower_data = [input_values[col] for col in feature_details.keys()]
        new_flower = np.array([new_flower_data])
        
        prediction_index = model.predict(new_flower)[0]
        predicted_species = target_names[prediction_index]
        
        icon = '🔴' if predicted_species == 'setosa' else \
               '🟡' if predicted_species == 'versicolor' else '🟣'
        
        st.markdown(f"""
        <div class="prediction-box">
            <p style="font-size: 24px; color: #4B0082;">Classification Successful!</p>
            <p class="prediction-text">{icon} The predicted species is: {predicted_species} {icon}</p>
        </div>
        """, unsafe_allow_html=True)
        
        input_df = pd.DataFrame([new_flower_data], columns=feature_details.keys())
        st.expander("View Input Data Used for Prediction").dataframe(input_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown('***')

# EDA AND VISUALIZATION SECTION 
st.markdown('### 2. Exploratory Data Analysis (EDA)')
st.markdown("Visualizing the dataset helps understand class separation and feature distributions.")

tab1, tab2 = st.tabs(["Species Distribution", "Feature Relationships (Pair Plot)"])

with tab1:
    st.subheader("Species Count Plot")
    fig_count, ax_count = plt.subplots(figsize=(6, 4), facecolor=APP_BACKGROUND_COLOR)
    sns.countplot(x='Species', data=df, ax=ax_count, palette='viridis')
    ax_count.set_title('Distribution of Iris Species')
    ax_count.set_xlabel('Species')
    ax_count.set_ylabel('Count')
    ax_count.set_facecolor('white')
    st.pyplot(fig_count)
    st.caption("The dataset is perfectly balanced with 50 samples for each of the three species.")

with tab2:
    st.subheader("Pair Plot")
    st.markdown("Shows the relationship between every pair of features, colored by species.")
    fig_pair = sns.pairplot(df, hue='Species', markers=["o", "s", "D"], corner=True)
    fig_pair.fig.patch.set_facecolor(APP_BACKGROUND_COLOR)
    plt.tight_layout()
    st.pyplot(fig_pair)
    st.caption("The visualizations clearly show that *Iris-setosa* is easily separable, while *versicolor* and *virginica* have some minor overlap.")

st.markdown('***')
st.caption('Developed as an Internship Project by Saanidhi Gade.')
