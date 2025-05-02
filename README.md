# 🏠 House Price Prediction

## 📘 Project Overview

This project aims to build a predictive model that estimates house sale prices based on various property features. The dataset is sourced from [Kaggle’s Ames Housing Dataset](https://www.kaggle.com/datasets). We created a fictitious use case in which predictive analytics are used to support real estate valuation for a property management platform.

Each row in the dataset represents a **single residential property**, and each column contains an attribute of that property. These attributes include **structural characteristics**, **quality assessments**, **construction details**, and **location-specific data**. The goal is to use these features to predict the `SalePrice` of a house.

This project demonstrates how **machine learning** can support **real estate professionals** and **property investors** in making data-informed decisions about pricing and valuation.

---

## 📊 Dataset Information

| Variable         | Description                                               | Unit / Range                         |
|------------------|-----------------------------------------------------------|--------------------------------------|
| 1stFlrSF         | First floor square footage                                | 334 - 4692 sq ft                     |
| 2ndFlrSF         | Second floor square footage                               | 0 - 2065 sq ft                       |
| BedroomAbvGr     | Bedrooms above ground level (excluding basement)          | 0 - 8                                |
| BsmtExposure     | Walkout or garden level walls exposure                    | Gd, Av, Mn, No, None                 |
| BsmtFinType1     | Basement finished area rating                             | GLQ, ALQ, BLQ, Rec, LwQ, Unf, None   |
| BsmtFinSF1       | Finished basement area (Type 1)                           | 0 - 5644 sq ft                       |
| BsmtUnfSF        | Unfinished basement area                                  | 0 - 2336 sq ft                       |
| TotalBsmtSF      | Total basement square footage                             | 0 - 6110 sq ft                       |
| GarageArea       | Garage size in square footage                             | 0 - 1418 sq ft                       |
| GarageFinish     | Interior finish of the garage                             | Fin, RFn, Unf, None                  |
| GarageYrBlt      | Year the garage was built                                 | 1900 - 2010                          |
| GrLivArea        | Above ground living area square footage                   | 334 - 5642 sq ft                     |
| KitchenQual      | Kitchen quality                                           | Ex, Gd, TA, Fa, Po                   |
| LotArea          | Lot size                                                  | 1300 - 215245 sq ft                  |
| LotFrontage      | Street frontage                                           | 21 - 313 linear feet                 |
| MasVnrArea       | Masonry veneer area                                       | 0 - 1600 sq ft                       |
| EnclosedPorch    | Enclosed porch area                                       | 0 - 286 sq ft                        |
| OpenPorchSF      | Open porch area                                           | 0 - 547 sq ft                        |
| OverallCond      | Overall condition of the house                            | 1 (Very Poor) - 10 (Very Excellent)  |
| OverallQual      | Overall material and finish of the house                  | 1 (Very Poor) - 10 (Very Excellent)  |
| WoodDeckSF       | Wood deck area                                            | 0 - 736 sq ft                        |
| YearBuilt        | Year of original construction                             | 1872 - 2010                          |
| YearRemodAdd     | Year of remodeling or addition                            | 1950 - 2010                          |
| SalePrice        | Sale price of the property                                | \$34,900 - \$755,000                |

---

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Seaborn & Matplotlib
- XGBoost
- Feature-engine
- Imbalanced-learn
- Plotly

---

## 🧠 Goals

- Perform EDA (Exploratory Data Analysis)
- Preprocess features and handle missing data
- Train regression models (Linear, Tree-based, Ensemble)
- Evaluate and tune model performance
- Deploy using Streamlit and Heroku

---
## 🧾 Project Terms & Jargon

| Term              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **Feature**        | An individual measurable property or variable in the dataset.              |
| **Target**         | The variable that we want to predict — in this case, `SalePrice`.          |
| **Encoding**       | Transforming categorical data into numerical form for modeling.            |
| **Null Values**    | Missing or empty data entries.                                             |
| **Pipeline**       | A series of data processing steps applied sequentially.                    |
| **Model Training** | The process of teaching a machine learning algorithm to make predictions.  |
| **Evaluation**     | Assessing the performance of the model using appropriate metrics.          |
| **Deployment**     | Making the trained model available to users via a web app or API.          |

---

## 💼 Business Requirements

The client has two primary business requirements:

1. **Correlation Insights**  
   The client is interested in understanding how various house attributes (features) correlate with the sale price. To support this, the project must provide:
   - Data visualisations of the most relevant features
   - Interpretations of how these features impact the final sale price

2. **Price Prediction**  
   The client has inherited four houses and wants to estimate their potential sale prices. Additionally, she wants the flexibility to predict the sale price of any house in Ames, Iowa based on its characteristics. The project must therefore include:
   - A trained machine learning model for price prediction
   - A user-friendly web interface (e.g., using Streamlit) for inputting house features and receiving price estimates
--- 
## 🔍 Hypothesis and Validation

This project is based on several key hypotheses regarding the factors that influence house sale prices in Ames, Iowa. Below are the primary hypotheses and the methods used to validate them:

### 📌 Hypotheses

1. **Larger houses sell for higher prices**  
   *Hypothesis:* Features like `GrLivArea`, `TotalBsmtSF`, and `GarageArea` have a positive correlation with `SalePrice`.

2. **Better quality and condition lead to higher prices**  
   *Hypothesis:* Variables such as `OverallQual`, `OverallCond`, and `KitchenQual` significantly influence the sale price.

3. **Renovated or newer homes are valued higher**  
   *Hypothesis:* `YearBuilt` and `YearRemodAdd` are positively correlated with `SalePrice`.

4. **Certain location-related attributes (e.g., `LotFrontage`, `LotArea`) impact price**  
   *Hypothesis:* Properties with more frontage or larger lots fetch higher sale prices.

5. **Basement and garage features add significant value**  
   *Hypothesis:* Finished basements (`BsmtFinType1`, `BsmtFinSF1`) and garage quality (`GarageFinish`, `GarageYrBlt`) positively affect the price.

### ✅ Validation Methods

- **Correlation Analysis**  
  Pearson correlation coefficients were calculated to identify relationships between numerical variables and `SalePrice`.

- **Data Visualizations**  
  Scatter plots, box plots, and heatmaps were used to visually assess relationships and distributions.

- **Feature Importance from ML Models**  
  Tree-based models (e.g., XGBoost) were used to identify which features most strongly impact the model's predictions.

- **Statistical Summaries**  
  Descriptive statistics and grouped analysis (e.g., average sale price by `OverallQual`) were performed to validate assumptions.

These hypotheses guided the feature selection, preprocessing, and model-building steps throughout the project.

---
## 💼 Machine Learning Business Case

The purpose of this project is to support the client in making informed, data-driven decisions about housing prices in Ames, Iowa. The client has inherited four houses and wants to understand their market value, as well as gain insights into factors that drive property prices in the area.

### 🎯 Objectives

1. **Price Prediction**  
   Build a machine learning model capable of predicting house sale prices based on various property features.

2. **Feature Insights**  
   Identify which house attributes most significantly impact the sale price to assist the client in evaluating or improving properties.

3. **Visualization for Understanding**  
   Provide clear visualisations that demonstrate correlations and trends in the housing data for easier decision-making.

### 🧠 ML Contribution to the Business

- **Valuation**: The ML model helps estimate fair market value for the client's properties without relying solely on real estate agents.
- **Strategic Renovations**: By identifying which features (e.g., kitchen quality, square footage) impact price the most, the client can prioritise renovation budgets.
- **Investment Analysis**: The tool can be generalised to assess other properties in Ames, enabling the client to expand or manage a real estate portfolio with confidence.

### 🤖 Model Selection

After performing exploratory data analysis and feature engineering, various machine learning models were tested, including:

- Linear Regression
- Random Forest

The model with the best performance based on RMSE and R² score was selected for final deployment.

---
## 🤖 Machine Learning Model Overview

### 🎯 Problem Statement

The objective is to build a regression model to accurately predict house **SalePrice** based on multiple property features. This will assist the client in pricing her inherited homes competitively and evaluating future investment opportunities in Ames, Iowa.

### 🧩 Model Type

We implement a **supervised regression model**, as the target variable (`SalePrice`) is continuous. The model is trained to learn relationships between housing features and their respective sale prices.

### 🔍 Target & Features

- **Target Variable**: `SalePrice`
- **Input Features**: All other relevant numeric and categorical variables from the dataset (e.g., `GrLivArea`, `OverallQual`, `GarageArea`, etc.)
- **Excluded Variables**: Irrelevant or high-missing-value features, or those causing data leakage.

### ⚙️ Data Source

The data used comes from the **Ames Housing Dataset** (publicly available via Kaggle), which includes approximately 2,900 property records and 80 features.

---

## 📏 Model Success Criteria

### ✅ Success Conditions

- **R² Score** of **≥ 0.70** on both training and test datasets.
- **RMSE (Root Mean Squared Error)** should be as low as possible, ideally under 30,000 USD.
- The model should generalize well without overfitting.

### ❌ Failure Conditions

- The model is deemed ineffective if its **R² Score < 0.60** on test data.
- If after several predictions, more than **30%** of predictions are off by **50% or more** compared to actual values.

---

## 🚀 Model Deployment Use Case

The model will be deployed via a **Streamlit web application**. Users can input property features manually or through forms, and receive real-time price predictions.

- **For the client**: Predict prices of inherited homes and understand the impact of various features.
- **For real estate analysts**: Generalize this model to evaluate other properties in Ames or similar markets.

---

## 🧠 Current Baseline & Heuristic

Before applying machine learning, there was no consistent method for estimating sale prices other than relying on agents or basic comparisons.

This model provides a more **data-driven**, **objective**, and **scalable** approach to valuation.

--- 

## 🖥️ Dashboard Design (Streamlit App User Interface)

The project includes a user-friendly **Streamlit dashboard** that provides a simple yet powerful interface for visualizing and interacting with the house price prediction model. The dashboard is designed with the goal of offering intuitive insights into the relationship between house attributes and predicted sale prices.

### 🎨 User Interface Components

The Streamlit app consists of several key components that allow the user to interact with the dataset and visualize important insights:

1. **Sidebar Navigation**  
   The sidebar allows users to navigate between different sections of the app, including:
   - **Home**: An overview of the project and key features.
   - **Data Exploration**: Visualizations of the dataset's most important features and their correlation with the sale price.
   - **Prediction**: A form where users can input house attributes to predict the sale price of a new property.

2. **Data Visualization**  
   The **Data Exploration** section includes the following:
   - **Scatter Plots**: Visualizations showing the relationship between features such as `GrLivArea`, `OverallQual`, and `SalePrice`.
   - **Correlation Heatmap**: A heatmap displaying the correlation matrix to highlight the most important variables affecting house prices.
   - **Box Plots**: For visualizing the distribution of house prices across categorical variables like `OverallCond`, `KitchenQual`, and others.

3. **Prediction Input Form**  
   In the **Prediction** section, users can input various house attributes (e.g., number of bedrooms, square footage, and condition) via an interactive form. Once the form is submitted:
   - The **predicted sale price** is displayed on the screen.
   - Key variables contributing to the predicted price are shown, providing insights into the model's decision-making process.

4. **Interactive Charts**  
   The dashboard includes dynamic, interactive charts powered by **Plotly** and **Seaborn**. These allow users to hover over and explore different data points for better understanding of how individual features impact the predictions.

5. **Model Performance Metrics**  
   The app also includes a section that displays important **model evaluation metrics** such as R², Mean Absolute Error (MAE), and Mean Squared Error (MSE) for both training and test sets, giving users insights into the model's performance.

### 🛠️ Streamlit Features Used

- **st.write()**: Used to display static text, images, and markdown.
- **st.sidebar**: Handles the sidebar elements for navigation and user input.
- **st.selectbox()** and **st.slider()**: Allow users to select features and input house characteristics interactively.
- **st.plotly_chart()**: Renders interactive Plotly visualizations for the user to explore data.
- **st.text_input()**: Captures user input for features such as `GrLivArea`, `OverallQual`, etc.

### 🧑‍💻 User Journey

1. **Data Exploration**:  
   Upon entering the dashboard, users are first greeted with an overview of the dataset through various visualizations. This section allows the user to explore the features and their relationships with the target variable (`SalePrice`).

2. **House Price Prediction**:  
   In the **Prediction** section, the user can input the features of a house (e.g., square footage, number of bedrooms, etc.), and the model will predict the **SalePrice** based on the provided input. The prediction is shown immediately along with a detailed breakdown of how each feature contributes to the prediction.

3. **Visualizing Results**:  
   After submitting the input, users can see the results visually displayed as graphs, and they can interact with these charts to further investigate how each feature affects the prediction.

### 💡 Future Enhancements

While the current dashboard provides essential functionality, there are several enhancements planned for future iterations, including:
- **Multiple Price Predictions**: Allowing users to input multiple house attributes at once and predict prices for a batch of houses.
- **Map Visualization**: Displaying house prices on an interactive map for users to understand the geographic distribution of house prices in Ames, Iowa.
- **Model Comparison**: Adding a feature to compare the performance of different models (e.g., Random Forest, Gradient Boosting) directly in the dashboard.

The **Streamlit app** serves as an interactive tool for users to easily explore the data, understand the relationships between house features and prices, and make predictions on new houses. It provides an accessible and engaging way to work with machine learning models in a real-world business context.





