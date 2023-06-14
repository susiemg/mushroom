import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load the dataset
data = pd.read_csv('mushrooms_clean.csv')

# Convert categorical variables to numeric using one-hot encoding
data_encoded = pd.get_dummies(data)

# Separate predictor variables (X) and response variable (y)
X = data_encoded.drop(['class_Edible','class_Poisonous'], axis=1)
y = data_encoded['class_Edible']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data into XGBoost's DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)

# Define the model parameters
params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}

# Train the model
model = xgb.train(params, dtrain)

# Convert the test set into DMatrix format
dtest = xgb.DMatrix(X_test)

# Predict the labels
y_pred = model.predict(dtest)

# Convert the predicted probabilities to binary labels
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

#saving the model
import pickle
pickle.dump(model,open("mushrooms_clf.pkl","wb"))
