import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib 

# --- 1. Load Pre-trained Model and Data ---
@st.cache_resource # Cache the loading to avoid re-loading on every rerun
def load_model_and_data():

    try:
        model = joblib.load('logistic_regression_breast_cancer_model.pkl')
        df_full = joblib.load('breast_cancer_data.pkl')
        feature_names = joblib.load('breast_cancer_feature_names.pkl')
        target_names = joblib.load('breast_cancer_target_names.pkl')
        st.success("Model and data loaded successfully!")
        return model, df_full, feature_names, target_names
    except FileNotFoundError:
        st.error(
            "Error: Model or data file not found. "
            "Please run 'python train.py' first to train and save the model."
        )
        st.stop() 

model, df_full, feature_names, target_names = load_model_and_data()



st.set_page_config(
    page_title="ML Model Deployment App",
    page_icon="🎗️", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Breast Cancer Diagnosis Prediction App")
st.markdown("---")

st.header("Input Tumor Measurements for Prediction")
st.write("Adjust the sliders to input values for various tumor characteristics.")

# Select a subset of features for user input to keep the UI manageable
# The model still uses all features for prediction.
input_features_for_ui = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness'
]

user_inputs = {}
for feature in feature_names:
    if feature in input_features_for_ui: # Only show sliders for selected features
        min_val = float(df_full[feature].min())
        max_val = float(df_full[feature].max())
        mean_val = float(df_full[feature].mean())
        user_inputs[feature] = st.slider(
            f"Select {feature.replace('_', ' ').title()}", # Clean up label
            min_value=min_val,
            max_value=max_val,
            value=mean_val, # Default to the mean
            step=(max_val - min_val) / 100 # Dynamic step for better control
        )
        st.write(f"You entered for {feature.replace('_', ' ').title()}: **{user_inputs[feature]:.2f}**")
    else:
        # For features not exposed via slider, use their mean value from the dataset
        user_inputs[feature] = float(df_full[feature].mean())

# Prepare input for prediction (ensure all original features are present)
input_for_prediction = pd.DataFrame([user_inputs])
# Ensure the order of columns matches the training data
input_for_prediction = input_for_prediction[feature_names]


st.markdown("---")
st.header("Model Prediction")

# Make Prediction
prediction_index = model.predict(input_for_prediction)[0]
predicted_diagnosis = target_names[prediction_index] # 0: malignant, 1: benign

# Get prediction probabilities
prediction_proba = model.predict_proba(input_for_prediction)[0]

# Display prediction with appropriate styling
if predicted_diagnosis == 'malignant':
    st.error(f"The predicted diagnosis is: **{predicted_diagnosis.upper()}**")
else:
    st.success(f"The predicted diagnosis is: **{predicted_diagnosis.upper()}**")

st.write("*(Prediction based on a Logistic Regression model trained on the Breast Cancer dataset)*")

st.subheader("Prediction Probabilities:")
proba_df = pd.DataFrame({
    'Diagnosis': target_names,
    'Probability': prediction_proba
}).sort_values(by='Probability', ascending=False)
st.dataframe(proba_df.style.format({'Probability': '{:.2%}'}), hide_index=True)


# --- 4. Visualization ---
st.markdown("---")
st.header("Model Output Visualization")
st.write("This plot shows the relationship between 'Mean Radius' and 'Mean Texture', with data points colored by actual diagnosis. Your input point and its predicted diagnosis are highlighted.")

fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for each class (0: malignant, 1: benign)
colors = {0: 'red', 1: 'green'} # Malignant in red, Benign in green

# Plot original data points, colored by actual diagnosis
for i, diagnosis_name in enumerate(target_names):
    subset = df_full[df_full['target'] == i]
    ax.scatter(
        subset['mean radius'],
        subset['mean texture'],
        color=colors[i],
        label=f'Actual {diagnosis_name.capitalize()}',
        alpha=0.6,
        s=50
    )

# Plot the user's input point
# Note: This visualization only uses 2 of the many input features for plotting.
# The prediction itself uses all features.
ax.scatter(
    user_inputs['mean radius'],
    user_inputs['mean texture'],
    color='purple',
    s=300,
    marker='*',
    label=f'Your Input (Predicted: {predicted_diagnosis.capitalize()})',
    edgecolor='black',
    linewidth=1,
    zorder=5
)

# Add annotation for the user's input
ax.annotate(
    f'Input:\nRadius:{user_inputs["mean radius"]:.2f}, Texture:{user_inputs["mean texture"]:.2f}\nPred: {predicted_diagnosis.capitalize()}',
    (user_inputs['mean radius'], user_inputs['mean texture']),
    textcoords="offset points",
    xytext=(10,10),
    ha='center',
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.7)
)


ax.set_xlabel("Mean Radius")
ax.set_ylabel("Mean Texture")
ax.set_title("Tumor Mean Radius vs. Mean Texture with Prediction")
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

st.pyplot(fig)

st.markdown("---")
st.info("This application now uses the Breast Cancer dataset for classification. The plot visualizes 'Mean Radius' vs. 'Mean Texture', highlighting your input and its predicted diagnosis.")
