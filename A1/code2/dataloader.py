
import torch
import pandas as pd
import numpy as np

class TrainDatasetReader(torch.utils.data.Dataset):

    def __init__(self, file_path: str):

        self.X = pd.read_csv(file_path, usecols = ['x1', 'x2'])
        self.y = pd.read_csv(file_path, usecols = ['label'])  
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # input : type List
        # output : type int
        input = self.X.loc[idx, :'x2'].values.astype(np.float32)
        output = self.y.loc[idx, 'label'].astype(np.float32)
        return input, output

class ValDatasetReader(torch.utils.data.Dataset):

    def __init__(self, file_path: str):
        self.X = pd.read_csv(file_path, usecols = ['x1', 'x2'])
        self.y = pd.read_csv(file_path, usecols = ['label'])  
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # input : type List
        # output : type int
        input = self.X.loc[idx, :'x2'].values.astype(np.float32)
        output = self.y.loc[idx, 'label'].astype(np.float32)
        return input, output

class TestDatasetReader(torch.utils.data.Dataset):

    def __init__(self, file_path: str):

        self.X = pd.read_csv(file_path, usecols = ['x1', 'x2'])
        self.y = pd.read_csv(file_path, usecols = ['label'])  
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # input : type List
        # output : type int
        input = self.X.loc[idx, :'x2'].values.astype(np.float32)
        output = self.y.loc[idx, 'label'].astype(np.float32)
        return input, output