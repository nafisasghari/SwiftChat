# Machine Learning Models
import pandas as pd
import numpy as np
import util

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import pickle


# Split Data
def split_df(df, test_size=0.2, random_state=None, drop_duplicate_train=False):
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    if drop_duplicate_train:
        train = util.drop_duplicates_rows(train)

    return train, test


def split_df_to_x_y(df, target, specify_features=None):
    """

    :param df:
    :param target: str, target name
    :param specify_features: list of features name
    :return:
    """
    if specify_features is None:
        features = [feat for feat in df.columns if feat != target]
    else:
        features = specify_features
    x = df[features]
    y = df[target]
    return x, y


def train_model_func(X, y, estimator, param_grid, cv=10, scoring='neg_log_loss'):
    grid_search = GridSearchCV(estimator, param_grid, cv=cv,
                               scoring=scoring)
    grid_search.fit(X, y)
    print("best parameters: ", grid_search.best_params_)
    print("-------" * 10)
    best_model = grid_search.best_estimator_
    print("best_model: ", best_model)

    return best_model

#Random Forest model

#xGboost model



def save_model(pkl_filename, model):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
