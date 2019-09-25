#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

def dataset(file_path):
    df = pd.read_csv(file_path)

    melbourne_features = ['Type','Method','Regionname','Rooms','Distance','Postcode','Bedroom2','Bathroom','Landsize','Lattitude','Longtitude','Propertycount']
    X = df[melbourne_features]
    y = df.Price

    return X, y

