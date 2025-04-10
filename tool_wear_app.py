import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)
n_samples = 200
cutting_speed = np.random.uniform(80, 200, n_samples)
feed_rate = np.random.uniform(0.1, 0.5, n_samples)
depth_of_cut = np.random.uniform(0.5, 3.0, n_samples)
machining_time = np.random.uniform(1, 60, n_samples)

# Tool wear formula (realistic simulation)
tool_wear = (
    0.002 * cutting_speed +
    0.5 * feed_rate +
    0.3 * depth_of_cut +
    0.05 * machining_time +
    np.random.normal(0, 0.2, n_samples)
)

# Create DataFrame
df = pd.DataFrame({
    "Cutting Speed (m/min)": cutting_speed,
    "Feed Rate (mm/rev)": feed_rate,
    "Depth of Cut (mm)": depth_of_cut,
    "Machining Time (min)": machining_time,
    "Tool Wear (mm)": tool_wear
})

# Split data
X = df.drop("Tool Wear (mm)", axis=1)
y = df["Tool Wear (mm)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.set_page_config(page_title="Tool Wear Predictor", layout="centered")
st.title("🛠️ Tool Wear Prediction System")
st.write("Enter the machining parameters to predict tool wear.")

# User input
cutting_speed_input = st.slider("Cutting Speed (m/min)", 80.0, 200.0, 120.0)
feed_rate_input = st.slider("Feed Rate (mm/rev)", 0.1, 0.5, 0.3)
depth_of_cut_input = st.slider("Depth of Cut (mm)", 0.5, 3.0, 1.5)
machining_time_input = st.slider("Machining Time (min)", 1.0, 60.0, 20.0)

input_data = pd.DataFrame({
    "Cutting Speed (m/min)": [cutting_speed_input],
    "Feed Rate (mm/rev)": [feed_rate_input],
    "Depth of Cut (mm)": [depth_of_cut_input],
    "Machining Time (min)": [machining_time_input]
})

# Predict and display
if st.button("Predict Tool Wear"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Tool Wear: {prediction:.3f} mm")

    # Visualize prediction vs true values
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7, color='teal')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel("Actual Tool Wear (mm)")
    ax.set_ylabel("Predicted Tool Wear (mm)")
    ax.set_title("Prediction vs Actual Tool Wear")
    st.pyplot(fig)

# Optional: show correlation heatmap
with st.expander("🔍 Show Data Correlation Heatmap"):
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

# Footer
st.markdown("---")
st.markdown("Built by Abhilash Mundepi   | Tool Wear Machine Learning Project")
