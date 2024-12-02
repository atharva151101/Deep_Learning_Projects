import os
import yaml
import torch
import numpy as np
import wandb
torch.manual_seed(42)

def accuracy(pred, target):
    pred_copy = np.array(pred)
    target_copy = np.array(target)
    #print(pred_copy)
    return (len(pred_copy[pred_copy == target_copy])/len(pred_copy))


def save_model(cfg, model):
    if not os.path.isdir(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)
    torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, 'models.ckpt'))



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
        print("gutyu ", target)
        target = target.type(torch.LongTensor)
        print("asafa ", target)
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


