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

## 📁 Directory Structure

