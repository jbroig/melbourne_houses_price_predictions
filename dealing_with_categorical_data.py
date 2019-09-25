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

# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print(object_cols)

# Comparing approaches to dealing with categorical variables

# Drop object columns (categorical variables)

drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

def score_dataset(X_train, X_valid, y_train, y_valid, method ):
    reg = RandomForestRegressor(n_estimators=100)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_valid)

    mae = mean_absolute_error(y_valid, y_pred)
    print('MAE using  {}: {:.0f}'.format(method, mae))

score_dataset(drop_X_train, drop_X_valid, y_train, y_valid, 'Drop categorial columns')

# Use Label Encoder

from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

label_encoder = LabelEncoder()

for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

score_dataset(label_X_train, label_X_valid, y_train, y_valid, 'Label encoder')

# Use One-Hot Encoder

'''
We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented in the training data, and
setting sparse=False ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).
'''

from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-Hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

score_dataset(OH_X_train, OH_X_valid, y_train, y_valid, 'One-Hot encoder')