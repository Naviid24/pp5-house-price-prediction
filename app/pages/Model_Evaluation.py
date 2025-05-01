import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# Load model and data
regression_model = joblib.load("models/random_forest_best_model.pkl")
df = pd.read_csv("data/processed/cleaned_corr_data.csv")

st.title("Model Evaluation")

# Prepare the data
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predictions
y_train_pred = regression_model.predict(X_train)
y_test_pred = regression_model.predict(X_test)

# R² Scores
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Display R² scores
st.subheader(f"Train R²: {r2_train:.3f}")
st.subheader(f"Test R²: {r2_test:.3f}")

# Plotting Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_train, y_train_pred, alpha=0.6, color='dodgerblue')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
axes[0].set_title(f"Train Set: Actual vs Predicted\nR² = {r2_train:.2f}")
axes[0].set_xlabel("Actual SalePrice")
axes[0].set_ylabel("Predicted SalePrice")

axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='forestgreen')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title(f"Test Set: Actual vs Predicted\nR² = {r2_test:.2f}")
axes[1].set_xlabel("Actual SalePrice")
axes[1].set_ylabel("Predicted SalePrice")

plt.tight_layout()
st.pyplot(fig)

# Business requirement comment
if r2_test >= 0.80:
    st.success("✅ The model meets the business requirement with an R² ≥ 0.80 on the test set.")
else:
    st.error("❌ The model does not meet the business requirement (R² < 0.80 on the test set).")
