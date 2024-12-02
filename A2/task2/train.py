from dataloader import DatasetReader1, DatasetReader2, DatasetReader3
from model import AE1,AE2,AE3,stacked_autoencoder
from encoder_functions import *
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
def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        print(name, params)
        total_params+=params
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__ == '__main__':

    cfg = open_config('config.yaml')

    # initialise wandb
    wandb.init(config = cfg, project='assignment2')
    cfg = wandb.config
        
    device = torch.device('cuda:0' if cfg.cuda else 'cpu')




    train_dataset = DatasetReader1(file_path = cfg.train_file_path, mode = 'train',num = cfg.ae1_input)
    val_dataset = DatasetReader1(file_path = cfg.val_file_path, mode = 'val',num = cfg.ae1_input)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=cfg.batch_size,
                                shuffle=True)

    valloader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=cfg.val_batch_size,
                            shuffle=True)  
    model1 = AE1(cfg)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model1.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    count_parameters(model1)
    best = 0
    for epoch in range(cfg.ae1_epochs):
        val_error = eval1(model1, 
                        valloader, criterion, device, epoch)
    
        train_one_epoch1(model1, trainloader, optimizer, 
                        criterion, epoch, device, cfg)
                        
    dataset = DatasetReader2(file_path = cfg.train_file_path,num = cfg.ae1_input)
    dataset = torch.utils.data.DataLoader(dataset, 
                                batch_size=1,
                                shuffle=True)                
    compress(model1.encoder1,cfg.ae1_output,"AE1_out.csv",dataset,device,cfg)    
    
    
    
    train_dataset = DatasetReader1(file_path = "AE1_out.csv", mode ='train',num = cfg.ae2_input)
    val_dataset = DatasetReader1(file_path = "AE1_out.csv", mode = 'val',num = cfg.ae2_input)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=cfg.batch_size,
                                shuffle=True)

    valloader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=cfg.val_batch_size,
                            shuffle=True)  
    model2 = AE2(cfg)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model2.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    count_parameters(model2)
    best = 0
    for epoch in range(cfg.ae2_epochs):
        val_error = eval1(model2, 
                        valloader, criterion, device, epoch)
    
        train_one_epoch1(model2, trainloader, optimizer, 
                        criterion, epoch, device, cfg)
                        
    dataset = DatasetReader2(file_path = "AE1_out.csv",num = cfg.ae2_input)
    dataset = torch.utils.data.DataLoader(dataset, 
                                batch_size=1,
                                shuffle=True)                
    compress(model2.encoder2,cfg.ae2_output,"AE2_out.csv",dataset,device,cfg)    
    
    
    
    
    train_dataset = DatasetReader1(file_path = "AE2_out.csv", mode ='train',num = cfg.ae3_input)
    val_dataset = DatasetReader1(file_path = "AE2_out.csv", mode = 'val',num = cfg.ae3_input)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=cfg.batch_size,
                                shuffle=True)

    valloader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=cfg.val_batch_size,
                            shuffle=True)  
    model3 = AE3(cfg)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model3.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    count_parameters(model3)
    best = 0
    for epoch in range(cfg.ae3_epochs):
        val_error = eval1(model3, 
                        valloader, criterion, device, epoch)
    
        train_one_epoch1(model3, trainloader, optimizer, 
                        criterion, epoch, device, cfg)
                        
    dataset = DatasetReader2(file_path = "AE2_out.csv",num = cfg.ae3_input)
    dataset = torch.utils.data.DataLoader(dataset, 
                                batch_size=1,
                                shuffle=True)                
     
    
    
    
    
    model4=stacked_autoencoder(cfg,model1,model2,model3)
    count_parameters(model4)  
    model4.load_state_dict(model1.state_dict(), strict=False)
    model4.load_state_dict(model2.state_dict(), strict=False)
    model4.load_state_dict(model3.state_dict(), strict=False)
    model4.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model4.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        
    train_dataset = DatasetReader3(file_path = cfg.train_file_path, mode = 'train')
    val_dataset = DatasetReader3(file_path = cfg.val_file_path, mode = 'val')
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=cfg.batch_size,
                                shuffle=True)

    valloader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=cfg.val_batch_size,
                            shuffle=True)
    print(model4.encoder1[0].weight)
    print(model1.encoder1[0].weight)
    best = 0
    for epoch in range(cfg.epochs):
        train_one_epoch2(model4, trainloader, optimizer, 
                        criterion, epoch, device, cfg)
        val_acc = eval2(model4, 
                    valloader, criterion, device, epoch)
    get_predictions(model4, valloader, "task_2_confusion_matrix", "")

