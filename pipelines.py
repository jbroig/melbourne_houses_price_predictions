#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from preparing_dataset import dataset


file_path = 'melb_data.csv'

X, y = dataset(file_path)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=1)

# Define preprocessing steps

'''
Similar to how a pipeline bundles together preprocessing and modeling steps, 
we use the ColumnTransformer class to bundle together different preprocessing steps. 

    - imputes missing values in numerical data
    - imputes missing values and applies a one-hot encoding to categorical data.
'''

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Get list of categorical variables
s = (X_train.dtypes == 'object')
categorical_cols = list(s[s].index)

# Get list of numerical variables (float + int)
f = (X_train.dtypes == 'float64')
i = (X_train.dtypes == 'int64')
float_cols = list(f[f].index)
int_cols = list(i[i].index)

numerical_cols = float_cols + int_cols


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle processing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers =[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)        
    ]
)

# Define the model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE: {:.0f}'.format(score))