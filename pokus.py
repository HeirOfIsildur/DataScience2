# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

col_target = 'booking_status'

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
    data = take_care_of_infinity(data)
    return data

def clean_data(data, col_target = col_target):
    data = data.drop(columns=['Booking_ID'])
    if col_target in data.columns:
        data = data.dropna(subset=[col_target], axis=0)

    categorical_columns = ['type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
                           'market_segment_type', 'repeated_guest']

    data = pd.get_dummies(data, columns=categorical_columns)
    return data

train = add_columns(train)
test_data = add_columns(test_data)

train = clean_data(train)
test_data = clean_data(test_data)

train.describe()
test_data.describe()

def get_numerical_predictors(data_: pd.DataFrame):
    cols = list(data_.columns)
    return [col for col in cols if data_[col].dtype != 'O']


def get_categorical_predictors(data_: pd.DataFrame):
    cols = list(data_.columns)
    return [col for col in cols if data_[col].dtype == 'O']

numerical_predictors = get_numerical_predictors(train)
categorical_predictors = get_categorical_predictors(train)


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

total_dr = np.mean(train[col_target])

for pred in categorical_predictors.copy():
    k = int(train[pred].nunique())

    if k <= 1:
        if pred in categorical_predictors:
            categorical_predictors.remove(pred)
            continue

    new_vals = mean_target_encoding(
            dt=train,
            predictor=pred,
            target=col_target
        )

    dt_grp = train.groupby(pred, dropna=False).agg(
            categ_dr = (col_target, np.mean),
            categ_cnt = (col_target, len)
        )

    additional_values = set(train[train[pred].notnull()][pred].unique()) - set(new_vals.keys())
    for p in additional_values:
        new_vals[p] = total_dr

    train['MTE_' + pred] = train[pred].replace(new_vals)
    test_data['MTE_' + pred] = test_data[pred].replace(new_vals)

train.drop(categorical_predictors, axis=1, inplace=True)
test_data = test_data.drop(categorical_predictors, axis=1, inplace=True)

#### In sample prediction
X = train.loc[:, train.columns != col_target ]
Y = train.loc[:, col_target]
model = XGBClassifier()
model.fit(X, Y)
prediction = model.predict(X)
accuracy = accuracy_score(prediction, Y)
print("Accuracy on training train: %.2f%%" % (accuracy * 100))

X = train.loc[:, train.columns != col_target ]
Y = train.loc[:, col_target]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=1)
model = XGBClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("Accuracy on testing data: %.2f%%" % (accuracy * 100))

# Let's see how many predictors we can omit
importances = list(model.feature_importances_)
features = list(model.feature_names_in_)

feature_importances = [(feature, importance) for feature, importance in zip(features, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Print out the feature and importances
print('feature : importance')
for feature, importance in feature_importances:
    print(f'{feature} : {"%0.2f" % importance} ')

features_sorted = [feature_importances[i][0] for i in range(len(feature_importances))]

"""
for i in range(len(features_sorted)):
    selected_features = features_sorted[: i + 1]
    X = train[selected_features]
    Y = train.loc[:, col_target]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    # fit model on all training train
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test train and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"For first {i + 1} most important features: Accuracy: %.2f%%" % (accuracy * 100))
"""
print()
print('As we can see, keeping 10 of the most important features might be optimal')
print('Which are these:')

the_important_features = features_sorted[:10]

X = train[the_important_features]
Y = train.loc[:, col_target]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
model = XGBClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("Accuracy on testing train using only 10 most important features: %.2f%%" % (accuracy * 100))


# Tune max_depth and min_child_weight.
from sklearn.model_selection import RandomizedSearchCV

# According to internet, parameters from those intervals are reasonable
param_grid = {
    'max_depth': [3, 5, 7, 9, 11, 13],
    'min_child_weight': [1, 3, 5],
    'gamma': [i / 10.0 for i in range(0, 10)],
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'n_estimators': [50 * i for i in range(1, 11)],
    'reg_alpha': [5e-2, 4e-2, 2e-2, 1e-2, 3e-2]
}

starting_grid = {
 'max_depth':[7],
 'min_child_weight':[1],
    'gamma': [8/10],
    'subsample': [9/10],
    'colsample_bytree': [7/10],
    'learning_rate': [0.05],
    'n_estimators': [150],
    'reg_alpha': [5e-2]
}


skutecny_grid = {
 'max_depth':7,
 'min_child_weight':1,
    'gamma': 8/10,
    'subsample': 9/10,
    'colsample_bytree': 7/10,
    'learning_rate': 0.05,
    'n_estimators': 150,
    'reg_alpha': 5e-2
}


X = train[the_important_features]
Y = train.loc[:, col_target]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
model = XGBClassifier(**skutecny_grid)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print("accuracy of the 1st fake grid is")
print(accuracy)

def change_param(data,new_data,param):
    data[param] = new_data[param]
    return data

best_params = {}
for param in list(param_grid):
    grid = change_param(starting_grid.copy(), param_grid.copy(), param)

    model = XGBClassifier()

    grid_search = GridSearchCV(model, grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)

    grid_search.fit(X, Y)

    best_params[param] = grid_search.best_params_[param]

X = train.loc[:, train.columns != col_target ]
Y = train.loc[:, col_target]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=1)
model = XGBClassifier(**best_params)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
print(f'starting accuracy is {accuracy}')
print(f'the optimal values for max depth and min child weight are {best_params}')


#nejlepsi model je
# {'max_depth': 9, 'min_child_weight': 1, 'gamma': 0.2, 'subsample': 0.8, 'colsample_bytree': 0.9, 'learning_rate': 0.1, 'n_estimators': 350, 'reg_alpha': 0.05}
# accuracy modelu = 0.755689


