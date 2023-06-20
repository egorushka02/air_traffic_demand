import pandas as pd

import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score, max_error




def get_error(model, X_valid, y_valid):
    preds = model.predict(X_valid)
    mse = mean_squared_error(preds, y_valid)
    r2 = r2_score(preds, y_valid)
    max_err = max_error(preds, y_valid)
    return mse, r2, max_err

def train_models(min_num_estimators, max_num_estimators, min_depth, max_depth):
    df = pd.read_excel("data/train.xlsx", sheet_name="Sheet1")
    #models = pd.DataFrame(columns = ['model_name', 'depth', 'num_estimators', 'mse', 'r2', 'max_error'])
    models=pd.DataFrame({'model_name':['model_1'], 'depth':[0], 'num_estimators':[0], 'mse':[0.0], 'r2':[0.0], 'max_error':[0.0]})
    X = df.drop(['RPK'], axis=1)
    y = df.RPK
    train_size = 0.8
    best_model = 'models/best_model.pkl'
    min_r_2 = -1

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = train_size, 
                                                      test_size = 1-train_size, random_state = 0)
    
    for depth in range(int(min_depth), int(max_depth)):
        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(X_train, y_train)
        errors = get_error(model, X_valid, y_valid)
        models.loc[len(models)] = [str(model), depth, 0, errors[0], errors[1], errors[2]]
        if errors[1] > min_r_2:
            with open(best_model, 'wb') as file:
                pickle.dump(model, file)

    for depth in range(int(min_depth), int(max_depth)):
        for estimators in range(int(min_num_estimators), int(max_num_estimators), 20):
            model = GradientBoostingRegressor(n_estimators = estimators, max_depth=depth)
            model.fit(X_train, y_train)
            errors = get_error(model, X_valid, y_valid)
            models.loc[len(models)] = [str(model), depth, estimators, errors[0], errors[1], errors[2]]
            if errors[1] > min_r_2:
                with open(best_model, 'wb') as file:
                    pickle.dump(model, file) 

    for depth in range(int(min_depth), int(max_depth)):
        model = DecisionTreeRegressor(max_depth=depth)
        model.fit(X_train, y_train)
        errors = get_error(model, X_valid, y_valid)
        models.loc[len(models)] = [str(model), depth, 0, errors[0], errors[1], errors[2]]
        if errors[1] > min_r_2:
            with open(best_model, 'wb') as file:
                pickle.dump(model, file)

    models.to_excel("statistics/stat.xlsx")
    