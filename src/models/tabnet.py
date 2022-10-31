import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.metrics import Metric

from ._models import rmse, RMSELoss

class TabNet:
    def __init__(self, args, data):
        super().__init__()
        
        self.criterion = RMSELoss
        self.data = data
        
        cat_dims = data['field_dims'].tolist()
        cat_idxs = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12]
        cat_emb_dim = args.EMBED_DIM
               
        
        self.batch_size = args.BATCH_SIZE
        self.epochs = args.EPOCHS
        #self.learning_rate = args.LR
        #self.weight_decay = args.WEIGHT_DECAY
        
        self.device = args.DEVICE
        self.model = TabNetRegressor(cat_dims=cat_dims, cat_idxs=cat_idxs, cat_emb_dim=cat_emb_dim, device_name=self.device)
    
    
    def train(self):
        X_train = self.data['X_train'].values
        y_train = self.data['y_train'].values.reshape(-1, 1)
        X_valid = self.data['X_valid'].values
        y_valid = self.data['y_valid'].values.reshape(-1, 1)
        # field dims
        # users     : 68069
        # isbn      : 149570
        # city      : 11989
        # state     : 1507
        # country   : 31
        # title     : 132713
        # author    : 59343
        # publisher : 968
        # category  : 152
        # cat_high  : 29
        # language  : 26
        self.model.fit(
            X_train, y_train,
            eval_set = [(X_valid, y_valid)],
            max_epochs = self.epochs, 
            eval_metric = ['rmse']
        )
    
    
    def predict_train(self):
        pass
    
    
    def predict(self, X_test):
        return self.model.predict(X_test)
