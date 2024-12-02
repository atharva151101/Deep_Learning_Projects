import os
import yaml
import torch
import numpy as np
import wandb
import pandas as pd
torch.manual_seed(42)

def accuracy(pred, target):
    pred_copy = np.array(pred)
    target_copy = np.array(target)
    #print(pred_copy)
    return (len(pred_copy[pred_copy == target_copy])/len(pred_copy))

def compress(model,num,path, loader,device,cfg):
    outputs=[]
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            outputs.append(np.append(output.numpy()[0],target.numpy())) 
            
    df=pd.DataFrame(outputs)
    cols={}
    for i in range(num):
        cols[i]=f'x{i}'
    cols[num]='y'    
    df=df.rename(columns=cols) 
    df.to_csv(path,header=True,index=False)


def train_one_epoch1(model, 
        dataset, 
        optimizer, 
        criterion, 
        epoch, 
        device,
        cfg):

    model.train()
    model.to(device)
    train_loss = []
    rolling_error = []
    
    for i, (data, target) in enumerate(dataset):
        #print("YES")
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output.squeeze(), target.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if (i==0):
            #print(target.detach().numpy(),output.squeeze().detach().numpy())
        diff = (target.squeeze() - output.squeeze()).detach().numpy()
        #print(diff.shape)
        mean_absolute_error = (np.sum(np.abs(diff))/diff.shape[0])/diff.shape[1]
        rolling_error.append(mean_absolute_error)
        train_loss.append(loss.item())

    wandb.log({
        'epoch': epoch,
        'train_loss':np.mean(train_loss),
        "train_mae": np.mean(rolling_error),
    })
    print(f'Epoch - {epoch}\t step - {i} \tTrain loss - {np.mean(train_loss)} \t MAE - {np.mean(rolling_error)}')

            
def eval1(model, valloader, criterion,device, epoch):

    val_acc = []
    model.eval()
    val_loss = []
    with torch.no_grad():
        for data, target in valloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            val_loss.append(loss.item())
            diff = (target - output.squeeze()).detach().numpy()
            val_acc.append(np.sum(np.abs(diff))/diff.shape[0])
    wandb.log({
        'epoch': epoch,
        'val_loss':np.mean(val_loss),
        "val_mae": np.mean(val_acc),
    })
            
    print(f"Val loss: {np.mean(val_loss)}\t Val error: {np.mean(val_acc)}")
    return np.mean(val_acc)

def train_one_epoch2(model, 
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
        optimizer.zero_grad()
        target = target.type(torch.LongTensor)
        #print("yoooo",output.squeeze().detach().numpy(),target.squeeze().detach().numpy())
        # output 32x1 -> 32
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



def eval2(model, valloader, criterion,device, epoch):

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


