#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from preparing_dataset import dataset


file_path = 'melb_data.csv'

X, y = dataset(file_path)

# Select numeric columns only
numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
X = X[numeric_cols]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=1)

from xgboost import XGBRegressor

'''
PARAMETERS 

- n_estimators: specifies how many times to go through the modeling cycle described above. 
    It is equal to the number of models that we include in the ensemble.

- early_stopping_rounds: offers a way to automatically find the ideal value for n_estimators. 
    Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. 
    It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.

- learning_rate: Instead of getting predictions by simply adding up the predictions from each component model, 
    we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.

-n_jobs: On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. 
    It's common to set the parameter n_jobs equal to the number of cores on your machine. On smaller datasets, this won't help.
    
'''

# Parameter tuning

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

predictions = my_model.predict(X_valid)

mae = mean_absolute_error(y_valid, predictions)

print('MAE: {:.0f}'.format(mae))