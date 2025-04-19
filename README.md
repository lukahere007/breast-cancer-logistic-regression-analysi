🧬 Breast Cancer Malignancy Prediction with Logistic Regression

This project develops a logistic regression model to predict breast cancer diagnosis (malignant vs benign) using features extracted from digitized images of fine needle aspirates (FNA) of breast masses. The dataset comes from the UCI Machine Learning Repository.
🔍 Project Overview

    Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set (n = 569)

    Objective: Use logistic regression to predict tumor malignancy

    Pipeline:

        Data visualization (ggplot2, ggstatsplot)

        Correlation and multicollinearity analysis

        Feature selection using AIC, deviance, accuracy, and AUC metrics

        Train-validation-test split (60%-20%-20%)

        Final model evaluation on the test set

✅ Final Model Summary

    Predictors: texture_mean, concavity_mean, radius_mean

    Formula:
    logit(p)=−35.05+0.404⋅texture_mean+29.67⋅concavity_mean+1.62⋅radius_mean
    logit(p)=−35.05+0.404⋅texture_mean+29.67⋅concavity_mean+1.62⋅radius_mean

    Test Set Performance:

        Accuracy: 90.3%

        Sensitivity: 92.96%

        Specificity: 85.71%

        AUC (ROC): 0.956

📊 Interactive Report

👉 RPubs Report [https://rpubs.com/lukahere007/breast-cancer-logistic-regression]
