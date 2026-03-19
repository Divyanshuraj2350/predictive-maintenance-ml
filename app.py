import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model = joblib.load("predictive_maintenance_model.pkl")

# Title
st.title("⚙️ Predictive Maintenance System")

st.write("Enter Machine Sensor Values")

# Input
col1, col2 = st.columns(2)

with col1:
    type_val = st.selectbox("Machine Type (0=L,1=M,2=H)", [0,1,2])
    air_temp = st.number_input("Air Temperature", value=298.0)
    process_temp = st.number_input("Process Temperature", value=308.0)

with col2:
    rpm = st.number_input("Rotational Speed", value=1500)
    torque = st.number_input("Torque", value=40)
    tool_wear = st.number_input("Tool Wear", value=5)

# PREDICTION
if st.button("Predict Failure"):

    data = pd.DataFrame([[

        type_val,
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear,
        0, 0, 0, 0, 0

    ]], columns=[
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "TWF","HDF","PWF","OSF","RNF"
    ])

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    failure_prob = probability[0][1]

    # Show probability
    st.metric("Failure Probability", f"{failure_prob*100:.2f}%")

    # Show result
    if prediction[0] == 1:
        st.error("⚠ Machine Failure Predicted")
    else:
        st.success("✅ Machine is Healthy")

# DIVIDER 
st.divider()

# FEATURE IMPORTANCE
st.subheader("📊 Model Feature Importance")

importance = model.feature_importances_

features = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

# Sort features for better visualization
indices = np.argsort(importance)

fig, ax = plt.subplots()

ax.barh(np.array(features)[indices], importance[indices])
ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")

st.pyplot(fig)
