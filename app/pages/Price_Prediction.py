import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained Random Forest model
model = joblib.load('models/random_forest_best_model.pkl')

# Load the cleaned data (your features should be based on this dataset)
df = pd.read_csv('data/processed/cleaned_corr_data.csv')

# Get the feature columns (drop the target variable 'SalePrice')
features = df.drop('SalePrice', axis=1).columns.tolist()

# Sidebar for user input
st.sidebar.header('Input House Features')

user_inputs = {}

# Determine min and max values for sliders based on your dataset
feature_ranges = {}
for feature in features:
    if feature not in ['KitchenQual', 'GarageFinish']:
        feature_ranges[feature] = {
            'min': int(df[feature].min()),
            'max': int(df[feature].max()),
            'default': int(df[feature].median())
        }

# Dynamically create sliders for the features
for feature in features:
    if feature not in ['KitchenQual', 'GarageFinish']:
        min_val = feature_ranges[feature]['min']
        max_val = feature_ranges[feature]['max']
        default_val = feature_ranges[feature]['default']
        user_inputs[feature] = st.sidebar.slider(
            feature, min_value=min_val, max_value=max_val, value=default_val
        )

# Explanation for KitchenQual
st.sidebar.subheader('Kitchen Quality')
st.sidebar.markdown("**0 is the worst, the highest number is the best!**")
user_inputs['KitchenQual'] = st.sidebar.selectbox(
    'Kitchen Quality (0 = Not Have , 1 = Fa , 2 = TA , 3 = Gd , 4 = Ex)', [
        0, 1, 2, 3, 4]
)

# Explanation for GarageFinish
st.sidebar.subheader('Garage Finish')
st.sidebar.markdown("**0 is the worst, the highest number is the best!**")
user_inputs['GarageFinish'] = st.sidebar.selectbox(
    'Garage Finish (0 = Not have , 1 = Unf , 2 = RFn , 3 = Fin)', [0, 1, 2, 3]
)

# Convert user inputs into a DataFrame for prediction
input_data = pd.DataFrame(user_inputs, index=[0])

# Ensure that input_data columns match the exact order
# and names used during training
input_data = input_data[features]

st.write(
    """
    For **GrLivArea, GarageArea, TotalBsmtSF** Unit / Range  is in sq ft
    """
    )

# Display user inputs for transparency
st.write("### User Inputs:")
st.write(input_data)

# Only check and show warning after the button is clicked
if st.button('Predict Sale Price'):
    # Check if all inputs are provided
    missing_inputs = [
        feature for feature, value in user_inputs.items() if value == 0
        ]  # Assuming 0 means missing value

    if missing_inputs:
        missing_features = ", ".join(missing_inputs)
        st.warning(
            f"Please fill in the following missing features: "
            f"{missing_features}."
        )
    else:
        try:
            prediction = model.predict(input_data)
            st.success(f"🏠 Predicted Sale Price: ${prediction[0]:,.2f}")
        except ValueError as e:
            st.error(f"Error during prediction: {str(e)}")


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption(
    "Created by [**Navid Bahadorani**](https://www.linkedin.com/in/navid-"
    "bahadorani-44a513299/) - 2025"
)
