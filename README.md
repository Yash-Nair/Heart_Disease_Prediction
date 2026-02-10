# Heart Disease Prediction using Machine Learning

## Overview
This project implements an end-to-end machine learning pipeline to predict the presence of heart disease using clinical patient data. The objective is to build and evaluate classification models while following proper machine learning practices such as data preprocessing, feature engineering, model comparison, and robust evaluation.

## Problem Statement
Heart disease is one of the leading causes of mortality worldwide. Early prediction using clinical indicators can assist healthcare professionals in identifying high-risk patients and enabling timely medical intervention. This project focuses on building a predictive model that classifies whether a patient is likely to have heart disease.

## Dataset
The dataset consists of patient health attributes such as age, sex, cholesterol levels, blood pressure, and other medical indicators commonly used in cardiovascular risk assessment.

## Project Workflow

1. Split the dataset into training and test sets  
2. Performed label encoding and checked for missing values and duplicate records  
3. Conducted Exploratory Data Analysis (EDA) to understand feature distributions and relationships  
4. Trained baseline classification models without optimization to establish initial performance benchmarks  

5. Performed hyperparameter tuning and probability threshold tuning using out-of-fold (OOF) predictions for the following classifiers:
   - Logistic Regression  
   - Support Vector Machine (SVM)  
   - K-Nearest Neighbors (KNN)  
   - Naive Bayes  
   - Decision Tree  
   - Random Forest  
   - Gradient Boosting  
   - AdaBoost  

   For each model:
   - Evaluated performance using the default classification threshold  
   - Optimized the decision threshold using OOF predictions  
   - Re-evaluated performance on the test dataset after threshold tuning  

6. Compared all models using classification metrics to identify the best-performing approach

## Models Implemented
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Gradient Boosting Classifier  
- Naive Bayes
- Adaboost

## Evaluation Metrics
Given the medical nature of the problem, evaluation was performed using multiple metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score
- Confusion Matrix

Special attention was given to recall and ROC-AUC to reduce false negatives, which is critical in healthcare applications.

## Results
The tuned model demonstrated improved performance in terms of ROC-AUC and recall, indicating its suitability for identifying patients at risk of heart disease while minimizing missed positive cases.

## Tools and Technologies
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Imbalanced-learn  
