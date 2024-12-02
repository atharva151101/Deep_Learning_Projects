from operator import mod
from dataloader import DatasetReader
from model import FunctionApprox
import torch.nn as nn
import torch.optim as optim
import os
import yaml
import torch
import numpy as np
import wandb
from plotting import plotting3D
torch.manual_seed(42)


def open_config(file):
    ''' Opens a configuration file '''
    config = yaml.safe_load(open(file, 'r'))
    return config

def save_model(cfg, model):
    if not os.path.isdir(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir)
    torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, 'models.ckpt'))

def load_model(cfg, model):
    assert cfg.model_path is not None
    model.load_state_dict(torch.load(cfg.model_path))
    return model

def get_plot(model,loader, name, title):
    model_outputs = []
    true_outputs = []
    x =[]
    y =[]
    with torch.no_grad():
        for data, target in loader:
            x.append(data.numpy()[0][0])
            y.append(data.numpy()[0][1])
            data = data.to(device)
            output = model(data)
            true_outputs.append(target.numpy())
            model_outputs.append(output.numpy()[0])

    plotting3D(np.array(x), 
                np.array(y), 
                np.array(model_outputs), 
                np.array(true_outputs),
                name = name,
                title= title)

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
    rolling_error = []
    
    for i, (data, target) in enumerate(dataset):
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        # output 32x1 -> 32
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()

        diff = (target - output.squeeze()).detach().numpy()
        mean_absolute_error = np.sum(np.abs(diff))/diff.shape[0]
        rolling_error.append(mean_absolute_error)
        train_loss.append(loss.item())

    wandb.log({
        'epoch': epoch,
        'train_loss':np.mean(train_loss),
        "train_mae": np.mean(rolling_error),
    })
    print(f'Epoch - {epoch}\t step - {i} \tTrain loss - {np.mean(train_loss)} \t MAE - {np.mean(rolling_error)}')

            
        
def eval(model, valloader, criterion,device, epoch):

    val_acc = []
    model.eval()
    val_loss = []
    with torch.no_grad():
        for data, target in valloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target)
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


if __name__ == '__main__':

    cfg = open_config('config.yaml')

    # initialise wandb
    wandb.init(config = cfg, project='assignment1', name = 'exp_model')
    cfg = wandb.config
    criterion = nn.MSELoss()  

    model = FunctionApprox(cfg)

    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
    device = torch.device('cuda:0' if cfg.cuda else 'cpu')

    train_dataset = DatasetReader(file_path = cfg.file_path, mode= 'train')

    trainloader = torch.utils.data.DataLoader(train_dataset, 
                            batch_size=cfg.batch_size,
                            shuffle=True)
    if cfg.do_val:
        val_dataset = DatasetReader(file_path = cfg.file_path, mode= 'val')
        

        valloader = torch.utils.data.DataLoader(val_dataset, 
                                batch_size=cfg.val_batch_size,
                                shuffle=True)
    test_dataset = DatasetReader(file_path = cfg.file_path, mode= 'test')
        

    testloader = torch.utils.data.DataLoader(val_dataset, 
                                batch_size=cfg.val_batch_size,
                                shuffle=True)
    if cfg.do_train:
        best = 1e100
        best_epoch = None
        for epoch in range(cfg.epochs):
            train_one_epoch(model, trainloader, optimizer, 
                            criterion, epoch, device, cfg)
            if cfg.do_val:
                val_error = eval(model, 
                        valloader, criterion, device, epoch)
            if val_error < best:
                best = val_error
                save_model(cfg, model)
                wandb.log({'val_mae_best': best})
                best_epoch = epoch
            if epoch in [1,2,10,50]:
                get_plot(model, trainloader, name = 'plot_epoch'+ str(epoch), title = 'Train')
        print(best_epoch)

    if cfg.do_predict:
        if not cfg.use_trained:
            model = FunctionApprox(cfg)
            model = load_model(cfg, model)
            model.eval()
        # using train_dataset
        get_plot(model, trainloader, name = 'q1_done', title= 'q1_done')


        # TODO add prediction code
    

    
    