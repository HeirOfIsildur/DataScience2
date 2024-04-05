import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('2024_DS2_HW1_data_train.csv')
test_data = pd.read_csv('2024_DS2_HW1_data_test.csv')

def add_no_of_all(data):
    data['no_of_all'] = data['no_of_adults']+data['no_of_children']
    return data

def add_no_of_nights(data):
    data['no_of_nights'] = data['no_of_weekend_nights']+data['no_of_week_nights']
    return data

def add_sin_cos_month(data):
    data['sin_month'] = np.sin(2*np.pi*data['arrival_date']/30)
    data['cos_month'] = np.cos(2*np.pi*data['arrival_date']/30)
    return data

def add_sin_cos_quarter(data):
    data['sin_quarter'] = np.sin(2*np.pi*data['arrival_date']/90)
    data['cos_quarter'] = np.cos(2*np.pi*data['arrival_date']/90)
    return data

def add_sin_cos_year(data):
    data['sin_year'] = np.sin(2*np.pi*data['arrival_date']/365)
    data['cos_year'] = np.cos(2*np.pi*data['arrival_date']/365)
    return data

def add_columns(data):
    data = add_no_of_all(data)
    data = add_no_of_nights(data)
    data = add_sin_cos_month(data)
    data = add_sin_cos_quarter(data)
    data = add_sin_cos_year(data)
    return data

train_data = add_columns(train_data)
test_data = add_columns(test_data)




#  # Get numerical feature importances
# importances = list(clf.feature_importances_)
# # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(predictors, importances)]
# # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# # Print out the feature and importances
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];