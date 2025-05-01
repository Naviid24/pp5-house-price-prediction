import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Set up the page configuration
st.set_page_config(page_title="House Price Prediction Dashboard", layout="wide")


# Title and introduction
st.title("🏡 Ames Housing Price Prediction Project")


st.markdown("""
### 🏡 Project Overview
Welcome to the **Ames House Price Prediction App**! 

The client is interested in discovering how house attributes correlate with sale price. 
To support this, we provide visualizations that show relationships between key variables and sale price.

Additionally, the client wants to predict the sale price of her four inherited houses, as well as any other house in Ames, Iowa, USA 🇺🇸.
This app allows users to explore the data and make predictions based on specific house features.
""")

# Key Features
st.header("✨ App Features")
st.markdown("""
- Predict house prices based on user inputs.
- Understand which house features (e.g., garage size, kitchen quality) most influence price.
- Explore the dataset with correlation heatmaps and feature importance visualizations.
- Compare different machine learning models' performances.
""")

# Add a callout/info box
st.info("""
💡 **Tip**: Use the sidebar to navigate between different sections: 
- Data Exploration
- Model Evaluation
- Price Classification
- Price Prediction

""")

# Footer
st.caption("Created with ❤️ by **Navid Bahadorani** - 2025")
