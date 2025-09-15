# Postpartum-depression-prediction-model
This project is built as a part of the requirements for partaking in the She Code Africa AI/ML Challenge.
Aim is to predict Post Partum Depression using survey and clinical data through regression modeling.

# Dataset
Source: https://drive.google.com/file/d/1b9479YMBAOlU-lIP0wL2wqouuxt1oYi6/view?usp=sharing
Dataset Schema: https://drive.google.com/file/d/1qGD13ErsDcvzwE4IFsWo30SaagZNBYFn/view?usp=sharing
Size: 1203, 394
Features: Mix of numeric, categorical, and boolean variables
Target variable: hamd_6m

# Methodology
- Data Cleaning
- Handled missing values (imputed numeric columns with median, categorical columns with mode).
- Replaced known “garbage values” and converted numeric-like text to numbers.
- Removed columns with >40% missing values.
- Checked and capped outliers using 1st and 99th percentile for continuous features.
- Corrected skewed distributions using Yeo-Johnson and log transformations.
- Feature Engineering
- Converted boolean columns to 0/1 integers.
- One-hot encoded categorical features.
- Removed identifier columns (newid, interviewer) to avoid bias.
- Feature Selection
- Removed highly correlated features (threshold = 0.8) while keeping the ones more correlated with the target.
- Scaling
- Standardized numeric features using StandardScaler.

# Model Training
Split data into train/test sets (80/20).
ElasticNet Regression with hyperparameter tuning (alpha and l1_ratio) using GridSearchCV.
Evaluated using R², MAE, RMSE.
> Best Model: ElasticNet with alpha=0.01 and l1_ratio=0.9
Performance: R² = 0.798, MAE = 0.314, RMSE = 0.448

# Visualizations
Predicted vs Actual: Shows how close predictions are to actual target values.
Residual Plot: Checks randomness of residuals to verify model assumptions.

# Saving & Deployment
Model saved as elasticnet_model.pkl
StandardScaler saved as scaler.pkl for future preprocessing of new data

# How to Run
1. Load elasticnet_model.pkl and scaler.pkl.
2. Preprocess your new dataset with the same cleaning, encoding, and scaling steps.
3. Use model.predict() to generate predictions.

# Notes
Only regression models were used per hackathon rules.
Steps were designed to handle unknown/hidden dataset variations without breaking.
Visualizations help check model fit and residuals

# Challenges:
- Data Cleaning
