import pandas as pd
import numpy as np
import warnings

import re
from pandas.api.types import CategoricalDtype
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split


from lightgbm import LGBMRegressor, LGBMClassifier, LGBMRanker
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

import wandb

from ._models import rmse, RMSELoss


class LGBM:
    def __init__(self, args, data):
        super().__init__()
        
        self.criterion = RMSELoss
        self.data = data
        
        self.params = args.PARAMS
        self.objective = 'rmse'
        
        self.device = args.DEVICE
        self.model = LGBMRegressor(**self.params, objective = self.objective, early_stopping_round=250)
    
    
    def train(self):
        print(self.data['X_train'].columns)
        X_train = self.data['X_train'].values
        y_train = self.data['y_train'].values
        X_valid = self.data['X_valid'].values
        y_valid = self.data['y_valid'].values
        
        self.model.fit(X_train, y_train, 
                       categorical_feature = [0, 1, 3, 4, 5, 7, 8, 9, 10, 11],
                       eval_set = [(X_valid, y_valid)])

    
    def predict(self, X_test):
        print(self.model.best_score_)
        return self.model.predict(X_test.values, num_iteration=self.model.best_iteration_)
    

class XGBoost:
    def __init__(self, args, data):
        super().__init__()
        
        self.data = data
        
        self.params = args.PARAMS
        self.objective = 'rmse'
        
        self.device = args.DEVICE
        self.model = XGBRegressor(**self.params, objective = self.objective, early_stopping_round=250)
    
    
    def train(self):
        print(self.data['X_train'].columns)
        X_train = self.data['X_train'].values
        y_train = self.data['y_train'].values
        X_valid = self.data['X_valid'].values
        y_valid = self.data['y_valid'].values
        
        self.model.fit(X_train, y_train, 
                       categorical_feature = [0, 1, 3, 4, 5, 7, 8, 9, 10, 11],
                       eval_set = [(X_valid, y_valid)])

    
    def predict(self, X_test):
        print(self.model.best_score_)
        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)


class CatBoost:
    def __init__(self, args, data):
        super().__init__()
        
        self.data = data
        
        self.params = args.PARAMS
        self.loss_function = 'RMSE'
        self.eval_metric = 'RMSE'
        
        self.device = args.DEVICE
        self.model = CatBoostRegressor(
            **self.params, 
            loss_function = self.loss_function, 
            eval_metric = self.eval_metric,
            task_type='GPU',
            verbose=True
        )
    
    
    def train(self):
        print(self.data['X_train'].columns)
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_valid = self.data['X_valid']
        y_valid = self.data['y_valid']
        
        self.model.fit(
            X_train, y_train, 
            cat_features = [0, 1, 3, 4, 5, 7, 8, 9, 10, 11],
            eval_set = [(X_valid, y_valid)],
            early_stopping_rounds=250
        )

    
    def predict(self, X_test):
        self.model.get_best_iteration()
        return self.model.predict(X_test)
