import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# Load the data
df = pd.read_csv("data/processed/cleaned_corr_data.csv")

# Load the trained Random Forest model
model = joblib.load('models/random_forest_best_model.pkl')

st.title("Data Exploration")

# Display summary stats
st.subheader('Dataset Overview')
st.write(df.describe())

# Correlation Heatmap
st.subheader('Correlation Heatmap')
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(plt)


# Get the feature columns (drop the target variable 'SalePrice')
features = df.drop('SalePrice', axis=1).columns.tolist()

# Feature Importance Visualization
st.subheader('Feature Importance')
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plotting the feature importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance for Predicting Sale Price')
st.pyplot(fig)


# SalePrice Distribution
st.subheader('Sale Price Distribution')
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, bins=30, color='green')
st.pyplot(plt)