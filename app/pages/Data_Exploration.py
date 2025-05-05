import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the data
df = pd.read_csv("data/processed/cleaned_corr_data.csv")

# Load the trained Random Forest model
model = joblib.load("models/random_forest_best_model.pkl")

# Page title
st.title("Data Exploration")

# ------------------------------
# Dataset Overview Explanation
# ------------------------------
st.markdown(
    """
    The **Dataset Overview** provides summary statistics for each feature
    in the dataset. These include **mean, standard deviation, minimum, and
    maximum values**, which give a sense of the central tendency and spread.
    This helps us understand the range and distribution of various features.
    """
)


# ------------------------------
# Display summary stats
# ------------------------------
st.subheader("Dataset Overview")
st.write(df.describe())

# ------------------------------
# Correlation Heatmap Section
# ------------------------------
st.markdown(
    """
    The **Correlation Heatmap** highlights how features relate to each
    other — especially their relationship with **SalePrice**. Strong
    positive or negative correlations indicate a potential impact on price.
    """
)


if st.checkbox("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

# ------------------------------
# Feature Importance Section
# ------------------------------
st.markdown(
    """
    **Feature Importance** shows the significance of each feature in predicting
    the target variable **SalePrice**. Features with higher importance values
    have a greater impact on predictions.
    """
)


features = df.drop("SalePrice", axis=1).columns.tolist()

if st.checkbox("Show Feature Importance"):
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"],
            importance_df["Importance"], color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance for Predicting Sale Price")
    st.pyplot(fig)

# ------------------------------
# Sale Price Distribution Section
# ------------------------------
st.markdown(
    """
    The **Sale Price Distribution** plot shows how the target variable
    **SalePrice** is spread. It helps us understand the range, skewness,
    and frequency of different price levels in the dataset.
    """
)


if st.checkbox("Show Sale Price Distribution"):
    st.subheader("Sale Price Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["SalePrice"], kde=True, bins=30, color="green")
    st.pyplot(plt)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption(
    "Created by [**Navid Bahadorani**](https://www.linkedin.com/in/navid-"
    "bahadorani-44a513299/) - 2025"
)
