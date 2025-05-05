import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Set page config
st.set_page_config(page_title="Inherited Houses Price", layout="wide")

# Title
st.title("🏠 Inherited Houses Price Prediction")

# Inherited houses dataset
df_inherited = pd.read_csv("data/raw/inherited_houses.csv")

# Select relevant columns
selected_columns = [
    'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF',
    'YearBuilt', 'YearRemodAdd', 'KitchenQual', 'GarageFinish'
]
df_selected = df_inherited[selected_columns]

# Show the selected data
st.subheader("📋 Selected Features of Inherited Houses")
st.markdown("These are the selected features that has the strong correlation with saleprice.")
st.dataframe(df_selected)

# Section title for predicted prices
st.subheader("💰 Predicted House Prices")
st.markdown("These are the estimated sale prices for four inherited houses based on their most effective features on SalePrice.")

# Predicted prices (with formatting for display)
predictions = {
    "House": ["First", "Second", "Third", "Fourth"],
    "Predicted Price": [124703.23, 148419.48, 170903.86, 185424.23]
}
df_prices = pd.DataFrame(predictions)

# Format price column with dollar signs
df_prices["Formatted Price"] = df_prices["Predicted Price"].apply(lambda x: f"${x:,.2f}")

# Show table if checked
if st.checkbox("Show Predicted Prices Table"):
    st.table(df_prices[["House", "Formatted Price"]])

# Chart section
if st.checkbox("Show Price Comparison Chart"):
    st.subheader("📊 Price Comparison Between Houses")
    st.markdown("Visual comparison of predicted prices for each house.")

    # Define custom order for chart
    order = ["First", "Second", "Third", "Fourth"]

    chart = alt.Chart(df_prices).mark_bar().encode(
        x=alt.X('House', sort=order, title='House'),
        y=alt.Y('Predicted Price', title='Price ($)', scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip('House'), alt.Tooltip('Predicted Price', format='$,.2f')]
    ).properties(
        width=600,
        height=400,
        title="Predicted Sale Price per House"
    )

    st.altair_chart(chart, use_container_width=True)


# Footer
st.markdown("---")
st.caption('Created by [**Navid Bahadorani**](https://www.linkedin.com/in/navid-bahadorani-44a513299/) - 2025')