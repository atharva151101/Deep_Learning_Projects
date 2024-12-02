
import torch
import pandas as pd
import random
import numpy as np
#from torchvision import transforms
from sklearn.model_selection import train_test_split


class DatasetReader1(torch.utils.data.Dataset):

    def __init__(self, file_path, mode):
        #self.transform = transforms.Compose([transforms.ToTensor()])
        self.cols = ["x{i}".format(i= i) for i in range(48)]
        X = pd.read_csv(file_path, usecols = self.cols)
        y = pd.read_csv(file_path, usecols = self.cols)
        y1= pd.read_csv(file_path, usecols = ['y'])
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, random_state = 42,stratify=y1)
       # X_val, Xtest, y_val, ytest = train_test_split(X_val, y_val, train_size = 0.01, random_state = 42)
        if mode == 'train':
            self.X = X_train.reset_index(drop = True)
            self.y = y_train.reset_index(drop = True)
        elif mode == 'val':
            self.X = X_val.reset_index(drop = True)
            self.y = y_val.reset_index(drop = True)
       
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # input : type List
        # output : type int
        input = self.X.loc[idx, :'x47'].values.astype(np.float32)
        output = input
        return input,output#self.transform(input), self.transform(output)
  
class DatasetReader2(torch.utils.data.Dataset):

    def __init__(self, file_path, mode ):
        self.cols = ["x{i}".format(i= i) for i in range(48)]
        X = pd.read_csv(file_path, usecols = self.cols)
        y = pd.read_csv(file_path, usecols = ['y'])

        self.X = X.reset_index(drop = True)
        self.y = y.reset_index(drop = True)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # input : type List
        # output : type int
        input = self.X.loc[idx, :'x47'].values.astype(np.float32)
        output = self.y.loc[idx, 'y'].astype(np.float32)
        return input, output
  
