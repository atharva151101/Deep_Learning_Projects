import os
import yaml
import numpy as np
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def open_config(file):
    ''' Opens a configuration file '''
    config = yaml.safe_load(open(file, 'r'))
    return config


if __name__ == '__main__':

    cfg = open_config('config.yaml')

    # initialise wandb
    wandb.init(config = cfg, project='assignment2')
    cfg = wandb.config
    cols = ["x{i}".format(i= i) for i in range(48)]
    X = pd.read_csv(cfg.train_file_path, usecols = cols)
    y = pd.read_csv(cfg.train_file_path, usecols = ['y'])
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, random_state = 42,stratify=y)
    #sc = StandardScaler()
 
    #X_train = sc.fit_transform(X_train)
    #X_val = sc.transform(X_val)
    pca = PCA(n_components = cfg.output_num)
 
    pca.fit(X_train)
    recon_train=pca.inverse_transform(pca.transform(X_train))
    recon_val = pca.inverse_transform(pca.transform(X_val))
    print(f'Train MAE - {np.mean(np.mean(np.abs(recon_train-X_train)))}\t Val MAE - {np.mean(np.mean(np.abs(recon_val-X_val)))}')
    
    out = pca.transform(X)
    
    df=pd.DataFrame(out)
    cols={}
    for i in range(10):
        cols[i]=f'x{i}'
    df=df.rename(columns=cols)
    df['y']=y 
    df.to_csv(cfg.out_file_path,header=True,index=False)

