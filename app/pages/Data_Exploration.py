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

# Add explanation before the dataset overview
st.markdown("""
    The **Dataset Overview** provides summary statistics for each feature in the dataset. 
    These statistics include measures such as **mean, standard deviation, minimum, and maximum values**, 
    which give us a sense of the central tendency and spread of the data. 
    This is crucial for understanding the range and distribution of values across various features.
""")

# Display summary stats
st.subheader('Dataset Overview')
st.write(df.describe())

# Add explanation before the correlation heatmap
st.markdown("""
    The **Correlation Heatmap** helps us understand the relationships between various features in the dataset, 
    particularly how each feature correlates with the target variable, **SalePrice**.
    Strong correlations may suggest that certain features are valuable predictors.
""")

# Correlation Heatmap (with checkbox for visibility)
if st.checkbox('Show Correlation Heatmap'):
    st.subheader('Correlation Heatmap')
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

# Feature Importance Visualization with explanation
st.markdown("""
    **Feature Importance** shows the significance of each feature in predicting the target variable **SalePrice**.
    Features with higher importance values have a greater impact on predictions.
""")

# Get the feature columns (drop the target variable 'SalePrice')
features = df.drop('SalePrice', axis=1).columns.tolist()

# Plotting the feature importances with checkbox for visibility
if st.checkbox('Show Feature Importance'):
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

# SalePrice Distribution with explanation
st.markdown("""
    The **Sale Price Distribution** plot shows the distribution of the target variable, **SalePrice**.
    It helps in understanding the range, skewness, and frequency of different price ranges in the dataset.
""")

# SalePrice Distribution (with checkbox for visibility)
if st.checkbox('Show Sale Price Distribution'):
    st.subheader('Sale Price Distribution')
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True, bins=30, color='green')
    st.pyplot(plt)


# Footer
st.markdown("---")
st.caption('Created by [**Navid Bahadorani**](https://www.linkedin.com/in/navid-bahadorani-44a513299/) - 2025')