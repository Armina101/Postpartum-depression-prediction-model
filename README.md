# Postpartum-depression-prediction-model
This project is built as a part of the requirements for partaking in the She Code Africa AI/ML Challenge.
Aim is to predict Post Partum Depression using the provided dataset through regression modeling.

# Dataset
- Source: https://drive.google.com/file/d/1b9479YMBAOlU-lIP0wL2wqouuxt1oYi6/view?usp=sharing
- Dataset Schema: https://drive.google.com/file/d/1qGD13ErsDcvzwE4IFsWo30SaagZNBYFn/view?usp=sharing
- Size: 1203, 394
- Features: Mix of numeric, categorical, and boolean variables
- Target variable: hamd_6m

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

# Model Training & Evaluation
- ElasticNet regression (with alpha=0.01, l1_ratio=0.9) was selected after testing Linear Regression and Ridge.
- Train/test split (80/20) allowed assessment of generalization.
- Performance metrics:
    - R² = 0.798: ~80% of variance in postpartum depression scores explained by the model—indicates strong predictive ability without overfitting.
    - MAE = 0.314: Average error in predictions is low, showing reliable estimates.
    - RMSE = 0.448: Confirms that extreme errors are small, consistent with MAE.

# Visualizations
- Actual vs Predicted plot: The points cluster around the diagonal red dashed line, indicating that the model’s predictions closely match the true target values. Most predictions are very near the actual postpartum depression scores, with minor deviations at the extremes, which shows strong model accuracy.
  
- Residuals vs Predicted plot: The residuals are scattered randomly around zero (red dashed line), with no obvious pattern or trend. This confirms that the model assumptions—linearity, homoscedasticity, and independence of errors—are reasonably satisfied. There are no systematic errors, suggesting that the ElasticNet model captures the relationships in the data well.

- Interpretation: Together, these plots validate that the model predictions are reliable and unbiased, with most errors being small and randomly distributed. This supports confidence in using the model for predictions on new data.

# Saving & Deployment
- Model saved as elasticnet_model.pkl
- StandardScaler saved as scaler.pkl for future preprocessing of new data

# How to Run
1. Load elasticnet_model.pkl and scaler.pkl.
2. Preprocess your new dataset with the same cleaning, encoding, and scaling steps.
3. Use model.predict() to generate predictions.

# Notes
- Only regression models were used per hackathon rules.
- Tested Linear Regression, Ridge, and ElasticNet; ElasticNet gave the best performance.
- Steps were designed to handle unknown/hidden dataset variations without breaking.
- Visualizations help check model fit and residuals.
