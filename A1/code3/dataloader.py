
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DatasetReader(torch.utils.data.Dataset):

    def __init__(self, file_path: str, mode: str):
        self.cols = [f'x{i}' for i in range(60)]
        X = pd.read_csv(file_path, usecols=self.cols)
        y = pd.read_csv(file_path, usecols=['y'])
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.7,random_state = 42, stratify = y)
        X_test, X_val, y_test, y_val = train_test_split(X_val, y_val, train_size = 0.66, random_state = 42, stratify =y_val )
        if mode == 'train':
            self.X = X_train.reset_index(drop = True)
            self.y = y_train.reset_index(drop = True)
        elif mode == 'val':
            self.X = X_val.reset_index(drop = True)
            self.y = y_val.reset_index(drop = True)
        elif mode == 'test':
            self.X = X_test.reset_index(drop = True)
            self.y = y_test.reset_index(drop = True)
  
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input = self.X.loc[idx, self.cols].values.astype(np.float32)
        output = self.y.loc[idx, 'y'].astype(np.float32)
        return input,output