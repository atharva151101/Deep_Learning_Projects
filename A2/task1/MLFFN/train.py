from dataloader import DatasetReader1
from model import MLFFN
from mlffn_functions import *
from utils import *
import torch.nn as nn
import torch.optim as optim
import os
import yaml
import torch
import numpy as np
import wandb
torch.manual_seed(42)

def open_config(file):
    ''' Opens a configuration file '''
    config = yaml.safe_load(open(file, 'r'))
    return config


if __name__ == '__main__':

    cfg = open_config('config.yaml')

    # initialise wandb
    wandb.init(config = cfg, project='assignment2')
    cfg = wandb.config
    
    criterion = nn.CrossEntropyLoss()  

    model = MLFFN(cfg)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        
    device = torch.device('cuda:0' if cfg.cuda else 'cpu')
	
    if cfg.use_autoencoder :
        path=cfg.autoencoder_compressed_path
    else : 
        path=cfg.pca_compressed_path
    
    train_dataset = DatasetReader1(file_path = path, mode = 'train')
    val_dataset = DatasetReader1(file_path = path, mode = 'val')
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=cfg.batch_size,
                                shuffle=True)

    valloader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=cfg.val_batch_size,
                            shuffle=True)

    best = 0
    for epoch in range(cfg.epochs):
        train_one_epoch2(model, trainloader, optimizer, 
                        criterion, epoch, device, cfg)
        val_acc = eval2(model, 
                    valloader, criterion, device, epoch)
        if val_acc > best:
            best = val_acc
            save_model(cfg, model)
            
    get_predictions(model,valloader,"plot2","")

