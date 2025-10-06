# Customer Churn Prediction Project

## Overview
This project implements a comprehensive Customer Churn Prediction system using machine learning. It includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Interactive Streamlit web application for predictions

## Dataset
The project uses the Telco Customer Churn dataset from Kaggle, which contains information about telecommunications customers and whether they churned.

## Project Structure
```
├── data/                          # Dataset files
├── notebooks/                     # Jupyter notebooks for analysis
├── models/                        # Trained model files
├── src/                          # Source code
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset:**
   - Sign up for Kaggle and get API credentials
   - Place kaggle.json in ~/.kaggle/
   - Run the data download script

3. **Run the analysis:**
   - Open and run the Jupyter notebooks in order
   - Train the models using the preprocessing pipeline

4. **Launch the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Features

### Data Analysis
- Comprehensive EDA with visualizations
- Customer behavior analysis
- Feature correlation analysis
- Data quality assessment

### Machine Learning
- Multiple algorithms: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with GridSearchCV
- Cross-validation and model evaluation
- Feature importance analysis

### Web Application
- Interactive form for customer data input
- Real-time churn prediction
- Model performance metrics display
- Feature importance visualization

## Model Performance
The models are evaluated using:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix

## Usage
1. Run the EDA notebook to understand the data
2. Execute the preprocessing and model training pipeline
3. Launch the Streamlit app for interactive predictions
4. Input customer characteristics to get churn predictions

## Dependencies
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning algorithms
- xgboost: Gradient boosting
- streamlit: Web application framework
- matplotlib, seaborn, plotly: Data visualization
- imbalanced-learn: Handling class imbalance
