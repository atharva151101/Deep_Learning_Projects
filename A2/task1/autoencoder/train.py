from dataloader import DatasetReader1, DatasetReader2
from model import AutoEncoder
from encoder_functions import *
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
    criterion = nn.L1Loss()  

    model = AutoEncoder(cfg)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        
    device = torch.device('cuda:0' if cfg.cuda else 'cpu')

    train_dataset = DatasetReader1(file_path = cfg.train_file_path, mode = 'train')
    val_dataset = DatasetReader1(file_path = cfg.val_file_path, mode = 'val')
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=cfg.batch_size,
                                shuffle=True)

    valloader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=cfg.val_batch_size,
                            shuffle=True)
   
    best = 0
    for epoch in range(cfg.epochs):
        val_error = eval1(model, 
                        valloader, criterion, device, epoch)
    
        train_one_epoch1(model, trainloader, optimizer, 
                        criterion, epoch, device, cfg)
        
    
    dataset = DatasetReader2(file_path = cfg.train_file_path, mode = 'train')
    dataloader=torch.utils.data.DataLoader(dataset, 
                                batch_size=1,
                                shuffle=False)

    compress(model,dataloader,device,cfg)

