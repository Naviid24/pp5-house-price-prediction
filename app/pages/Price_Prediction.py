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

# Dynamically create input fields for the other features
for feature in features:
    if feature not in ['KitchenQual', 'GarageFinish']:  # Skip the ones already handled
        user_inputs[feature] = st.sidebar.number_input(
            feature, min_value=0, value=0, step=1
        )

# Explanation for KitchenQual
st.sidebar.subheader('Kitchen Quality')
st.sidebar.markdown("**0 is the worst, the highest number is the best!**")
user_inputs['KitchenQual'] = st.sidebar.selectbox(
    'Kitchen Quality (0 = Not Have , 1 = Fa , 2 = TA , 3 = Gd , 4 = Ex)', [0, 1, 2, 3, 4]
)

# Explanation for GarageFinish
st.sidebar.subheader('Garage Finish')
st.sidebar.markdown("**0 is the worst, the highest number is the best!**")
user_inputs['GarageFinish'] = st.sidebar.selectbox(
    'Garage Finish (0 = Not have , 1 = Unf , 2 = RFn , 3 = Fin)', [0, 1, 2, 3]
)

# Convert user inputs into a DataFrame for prediction
input_data = pd.DataFrame(user_inputs, index=[0])


# Ensure that input_data columns match the exact order and names used during training
input_data = input_data[features]

# Display user inputs for transparency
st.write("### User Inputs:")
st.write(input_data)

# Only check and show warning after the button is clicked
if st.button('Predict Sale Price'):
    # Check if all inputs are provided
    missing_inputs = [feature for feature, value in user_inputs.items() if value == 0]  # Assuming 0 means missing value
    
    if missing_inputs:
        # Show the warning message only if there are missing inputs
        missing_features = ", ".join(missing_inputs)
        st.warning(f"Please fill in the following missing features: {missing_features}.")
    else:
        # If no inputs are missing, proceed with prediction
        try:
            prediction = model.predict(input_data)
            st.success(f"🏠 Predicted Sale Price: ${prediction[0]:,.2f}")
        except ValueError as e:
            st.error(f"Error during prediction: {str(e)}")



