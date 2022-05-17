# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:02:12 2022

@author: aceso
"""

#%% Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import missingno as msno
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#%% Constants
DATA_PATH = os.path.join(os.getcwd(), "Data","heart.csv")
SCALER_SAVEPATH = os.path.join(os.getcwd(), "Saved", "minmax.pkl")
MODEL_PATH = os.path.join(os.getcwd(), "Saved", "model.pkl")

#%% EDA
df = pd.read_csv(DATA_PATH)
print(df.head())

#%% Data Inspection
df.info()
df.describe().T
df.boxplot()
df.duplicated().sum() # 1 duplicate

#%% Data Cleaning
# Drop duplicate
clean_data = df.drop_duplicates()
clean_data.info()
clean_data.duplicated().sum()

# Check NaN value
msno.matrix(clean_data)
clean_data.isnull().sum() # no NaN value

#%% Feature Selection
# Using Lasso Regression
X = clean_data.drop(labels=["output"], axis=1)
y = np.expand_dims(clean_data["output"], axis=-1)

# Visualize
cor = df.corr() # correllate each columns

# Plot a figure
plt.figure(figsize=(12,10))
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Since all features are below 50% correlation to the labels
# I decided to select all the features into training

#%% Feature Scaling
# I decided to use min max scalling since there's no -ve value in the features
minmax = MinMaxScaler()
clean_data = minmax.fit_transform(X)
sns.distplot(clean_data)
plt.title("Min Max Scalling Data")
plt.legend()
plt.show()

# Data already in uniform scale from 0 to 1

# Saving the scale
pickle.dump(minmax, open(SCALER_SAVEPATH, "wb"))

#%% Data Preprocessing 

# Split the data into training and test data


#%% Model Configuration
# I'm using Random Forest Classification for this problem
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)

# Instantiate the classifier and fit the training data
forest = RandomForestClassifier().fit(X_train, y_train)

#%% Model Evaluation
acc = forest.score(X_test, y_test)
print(f"The accuracy for this estimator is {acc:.00%}")

#%% Model Saving

model = pickle.dump(forest, open(MODEL_PATH, "wb"))
