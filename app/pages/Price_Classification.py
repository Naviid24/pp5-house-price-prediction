
import streamlit as st
import pandas as pd
import pickle
import numpy as np


# ------------------------------
# Load the pre-trained model
# ------------------------------
with open('models/random_forest_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ------------------------------
# Pafe Configuration
# ------------------------------
st.title('🏡 Ames House Price Category Prediction')

st.markdown(
    """
    This app predicts the **price category** (Low, Medium, High)
    of a house in Ames, Iowa based on several features.
    """
    )


# ------------------------------
# Input fields for user data
# ------------------------------
st.sidebar.header('Input Features')


def user_input_features():
    # You can add any features here that are relevant to your model
    overall_qual = st.sidebar.slider(
        'Overall Quality (OverallQual)', 1, 10, 5
        )
    gr_liv_area = st.sidebar.slider(
        'Above Ground Living Area (GrLivArea)', 500, 5000, 1500
        )
    garage_area = st.sidebar.slider(
        'Garage Area (GarageArea)', 0, 1500, 500
        )
    total_bsmt_sf = st.sidebar.slider(
        'Total Basement Area (TotalBsmtSF)', 0, 5000, 1000
        )
    year_built = st.sidebar.slider(
        'Year Built (YearBuilt)', 1900, 2022, 2000
        )
    year_remod_add = st.sidebar.slider(
        'Year Remodeled (YearRemodAdd)', 1900, 2022, 2000
        )
    kitchen_qual = st.sidebar.selectbox(
        'Kitchen Quality (KitchenQual)', ['Ex', 'Gd', 'TA', 'Fa', 'Po']
        )
    garage_finish = st.sidebar.selectbox(
        'Garage Finish (GarageFinish)', ['Fin', 'RFn', 'Unf', 'NoGarage']
        )

    data = {
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'GarageArea': garage_area,
        'TotalBsmtSF': total_bsmt_sf,
        'YearBuilt': year_built,
        'YearRemodAdd': year_remod_add,
        'KitchenQual': kitchen_qual,
        'GarageFinish': garage_finish,
    }

    features = pd.DataFrame(data, index=[0])
    return features


# ------------------------------
# User Inputs
# ------------------------------
input_data = user_input_features()

st.subheader('User Input Features')
st.write(input_data)

# ------------------------------
# Encode categorical features if needed
# ------------------------------
input_data_encoded = input_data.copy()
input_data_encoded['KitchenQual'] = input_data_encoded['KitchenQual'].map(
    {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    )
input_data_encoded['GarageFinish'] = input_data_encoded['GarageFinish'].map(
    {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NoGarage': 0}
    )

# ------------------------------
# Make prediction
# ------------------------------
prediction = model.predict(input_data_encoded)
st.subheader('Prediction Result')
st.write(f"The predicted price category is: **{prediction[0]}**")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption(
    "Created by [**Navid Bahadorani**](https://www.linkedin.com/in/navid-"
    "bahadorani-44a513299/) - 2025"
)
