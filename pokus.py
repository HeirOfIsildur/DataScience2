# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# load the data
train = pd.read_csv('2024_DS2_HW1_data_train.csv')
test_data = pd.read_csv('2024_DS2_HW1_data_test.csv')


def plot_hist(data):
    for column in data.columns:
        data[column].hist()
        plt.title(column)
        plt.show()
    return


# add new columns
def add_columns(data):
    def add_no_of_all(data):
        data['no_of_all'] = data['no_of_adults'] + data['no_of_children']
        return data

    def add_no_of_nights(data):
        data['no_of_nights'] = data['no_of_weekend_nights'] + data['no_of_week_nights']
        return data

    def add_sin_cos_month(data):
        data['sin_month'] = np.sin(2 * np.pi * data['arrival_date'] / 30)
        data['cos_month'] = np.cos(2 * np.pi * data['arrival_date'] / 30)
        return data

    def add_sin_cos_quarter(data):
        data['sin_quarter'] = np.sin(2 * np.pi * data['arrival_date'] / 90)
        data['cos_quarter'] = np.cos(2 * np.pi * data['arrival_date'] / 90)
        return data

    def add_sin_cos_year(data):
        data['sin_year'] = np.sin(2 * np.pi * data['arrival_date'] / 365)
        data['cos_year'] = np.cos(2 * np.pi * data['arrival_date'] / 365)
        return data

    def add_year_month(data):
        data['year_month'] = data['arrival_year'] * 100 + data['arrival_month']
        return data

    def take_care_of_infinity(data):
        data = data.replace([np.inf, -np.inf], np.nan)
        return data

    data = add_no_of_all(data)
    data = add_no_of_nights(data)
    data = add_sin_cos_month(data)
    data = add_sin_cos_quarter(data)
    data = add_sin_cos_year(data)
    data = add_year_month(data)
    return data

def clean_data(data):
    def check_unique(data):
        for column in data.columns:
            if len(data[column].unique()) == 1:
                print(column)
        return

    # check if there are rows full of nans
    def check_nan(data):
        for index, row in data.iterrows():
            if row.isnull().all():
                print(index)
        return

    check_unique(train)
    check_unique(test_data)

    check_nan(train)
    check_nan(test_data)

    if 'booking_status' in data.columns:
        data = data.dropna(subset=['booking_status'], axis=0)

    categorical_columns = ['Booking_ID', 'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
                           'market_segment_type', 'repeated_guest']

    data = pd.get_dummies(data, columns=categorical_columns)
    return data

train = add_columns(train)
test_data = add_columns(test_data)

train = clean_data(train)
test_data = clean_data(test_data)

train_data = pd.get_dummies(train)
test_data = pd.get_dummies(test_data)

# plot_hist(train)
# plot_hist(test_data)

train_data = train.drop(['booking_status'], axis=1)
train_labels = train['booking_status']

# Initialize the model
model = XGBClassifier()

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Create a GridSearchCV instance
grid_search = GridSearchCV(model, param_grid, cv=10, verbose=3)

# Perform the grid search and fit the model with the best found parameters
grid_search.fit(train_data, train_labels)

# Print the best parameters and the score of the best model
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

#  # Get numerical feature importances
# importances = list(clf.feature_importances_)
# # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(predictors, importances)]
# # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# # Print out the feature and importances
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


def mean_target_encoding(dt, predictor, target, alpha=0.01):
    total_cnt = len(dt)
    total_dr = np.mean(dt[target])
    dt_grp = dt.groupby(predictor).agg(
        categ_dr=(target, np.mean),
        categ_cnt=(target, len)
    )

    dt_grp['categ_freq'] = dt_grp['categ_cnt'] / total_cnt
    dt_grp['categ_encoding'] = (dt_grp['categ_freq'] * dt_grp['categ_dr'] + alpha * total_dr) / (
                dt_grp['categ_freq'] + alpha)

    return dt_grp[['categ_encoding']].to_dict()['categ_encoding']