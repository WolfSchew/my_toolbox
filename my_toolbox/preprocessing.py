import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

def get_data(df, target_column):
    X = df.drop(columns=target_column)
    y = df[target_column]

    return X,y


def quick_ohe(df, column):
    encoder = OneHotEncoder(sparse=False)
    encoded_columns = encoder.fit_transform(df[column])

    return encoded_columns

def quick_ohe_binary(df, column):
    encoder = OneHotEncoder(drop='if_binary', sparse=False)
    df[column] = encoder.fit_transform(df[column])

    return df