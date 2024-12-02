from dataloader import DatasetReader
from model import ClassificationImage
import torch.nn as nn
import torch.optim as optim
import os
import yaml
import torch
import numpy as np
import wandb
from utils import *
import warnings
warnings.filterwarnings("ignore")


def open_config(file):
    ''' Opens a configuration file '''
    config = yaml.safe_load(open(file, 'r'))
    return config

def accuracy(pred, target):
    pred_copy = np.array(pred)[0]
    target_copy = np.array(target)[0]
    return (1 if pred_copy == target_copy else 0)


def save_model(cfg, model):
    if not os.path.isdir(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)
    torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, 'models.ckpt'))

def train_one_epoch(model, 
        dataset, 
        optimizer, 
        criterion, 
        epoch, 
        device,
        cfg):

    model.train()
    model.to(device)
    train_loss = []
    acc = []
    
    for i, (data, target) in enumerate(dataset):
        data, target = data.to(device), target.to(device)
        output = model(data)
        print(output.detach().numpy(),target.detach().numpy())	
        optimizer.zero_grad()
        target = target.type(torch.LongTensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        idxs = torch.argmax(output, dim = 1)
        acc.append(accuracy(idxs, target))   
        train_loss.append(loss.item())
        
    wandb.log({
        'epoch': epoch,
        'train_loss':np.mean(train_loss),
        "train_acc": np.mean(acc),
    })
    print(f'Epoch - {epoch}\t step - {i} \tTrain loss - {np.mean(train_loss)} \t ACC - {np.mean(acc)}')

            
        
def eval(model, valloader, criterion,device, epoch):

    val_acc = []
    model.eval()
    val_loss = []
    with torch.no_grad():
        for data, target in valloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.type(torch.LongTensor)
            loss = criterion(output, target)
            val_loss.append(loss.item())
            idxs = torch.argmax(output, dim = 1)
            val_acc.append(accuracy(idxs, target))   
    wandb.log({
        'epoch': epoch,
        'val_loss':np.mean(val_loss),
        "val_acc": np.mean(val_acc),
    })
            
    print(f"Val loss: {np.mean(val_loss)}\t Val acc: {np.mean(val_acc)}")
    return np.mean(val_acc)


if __name__ == '__main__':

    cfg = open_config('config.yaml')
    torch.manual_seed(42)

    # initialise wandb
    wandb.init(config = cfg, project='assignment1', name = f"code3_{cfg['optim']}")
    cfg = wandb.config
    criterion = nn.CrossEntropyLoss()  
    model = ClassificationImage(cfg)
    if cfg.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
    elif cfg.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optim == 'delta':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate)
        
    device = torch.device('cuda:0' if cfg.cuda else 'cpu')

    train_dataset = DatasetReader(file_path = cfg.train_file_path, mode = 'train')
    val_dataset = DatasetReader(file_path = cfg.train_file_path, mode = 'val')
    # PATTERN MODE by setting batch_size = 1
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=cfg.batch_size,
                                shuffle=True)
    # PATTERN MODE by setting batch_size = 1
    valloader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=cfg.val_batch_size,
                            shuffle=True)

    best = 0
    best_epoch = None
    for epoch in range(cfg.epochs):
        train_one_epoch(model, trainloader, optimizer, 
                        criterion, epoch, device, cfg)
        val_acc = eval(model, 
                    valloader, criterion, device, epoch)
        if val_acc > best:
            best = val_acc
            best_epoch =epoch
            save_model(cfg, model)
            wandb.log({'val_acc_best': best})
    print(f'The best model checkpoint was saved on {best_epoch} epoch')

    # load best checkpoint
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_dir, 'models.ckpt')))

    # confusion matix
    y_tue = []
    y_ped = []
    with torch.no_grad():
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_tue.append(target)
            y_ped.append(torch.argmax(output).item())
    get_matrix(y_ped, y_tue, name = f'tain_{cfg.optim}')

    test_dataset = DatasetReader(file_path = cfg.train_file_path, mode = 'test')
    
    testloader = torch.utils.data.DataLoader(test_dataset, 
                                batch_size=cfg.batch_size,
                                shuffle=True)
    y_tue = []
    y_ped = []
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            output = model(data)
            y_ped.append(torch.argmax(output).item())
    

        
