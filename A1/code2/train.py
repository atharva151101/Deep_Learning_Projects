from dataloader import ValDatasetReader, TrainDatasetReader
from model import Classification2D
import torch.nn as nn
import torch.optim as optim
import os
import yaml
import torch
import numpy as np
import wandb
from plotting import plot_decision_boundary
torch.manual_seed(42)


def get_plot(model,loader, name, title):
    model_outputs = []
    true_outputs = []
    x1 =[]
    x2 =[]
    with torch.no_grad():
        for data, target in loader:
            x1.append(data.numpy()[0][0])
            x2.append(data.numpy()[0][1])
            data = data.to(device)
            output = model(data)
            true_outputs.append(target.numpy()[0])
            model_outputs.append(output.numpy()[0])    
        x1=np.array(x1)
        x2=np.array(x2)
        model_outputs=np.array(model_outputs)
        true_outputs=np.array(true_outputs)
        model_outputs[model_outputs>0.5]=1
        model_outputs[model_outputs<=0.5]=0
        min1=x1.min()-0.5
        max1=x1.max()+0.5
        min2=x2.min()-0.5
        max2=x2.max()+0.5
        x1_mesh = np.arange(min1, max1, 0.01)
        x2_mesh = np.arange(min2, max2, 0.01)
        xx, yy = np.meshgrid(x1_mesh, x2_mesh)
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        grid = np.float32(np.hstack((r1,r2)))
        yhat=model(torch.from_numpy(grid)).numpy()
        zz = yhat.reshape(xx.shape)
        zz[zz>0.5]=1
        zz[zz<=0.5]=-1
        plot_decision_boundary(x1, x2, true_outputs,xx,yy,zz)
	
def open_config(file):
    ''' Opens a configuration file '''
    config = yaml.safe_load(open(file, 'r'))
    return config

def accuracy(input, target):
    assert len(input) == len(target)
    return 1 - (np.abs(input - target).sum())/len(input)


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
    rolling_error = []
    
    for i, (data, target) in enumerate(dataset):
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(output.squeeze(0), target)
        loss.backward()
        optimizer.step()
        classes = [1 if i > 0.5 else 0 for i in output.detach().numpy()] 
        acc = accuracy(classes, target.numpy())
        rolling_error.append(acc) 
        train_loss.append(loss.item())
        
    wandb.log({
        'epoch': epoch,
        'train_loss':np.mean(train_loss),
        "train_acc": np.mean(rolling_error),
    })
    print(f'Epoch - {epoch}\t step - {i} \tTrain loss - {np.mean(train_loss)} \t ACC - {np.mean(rolling_error)}')

            
        
def eval(model, valloader, criterion,device, epoch):

    val_acc = []
    model.eval()
    val_loss = []
    with torch.no_grad():
        for data, target in valloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(0), target)
            val_loss.append(loss.item())
            classes = [1 if i > 0.5 else 0 for i in output.numpy()] 
            val_acc.append(accuracy(classes, target.numpy())) 
    wandb.log({
        'epoch': epoch,
        'val_loss':np.mean(val_loss),
        "val_acc": np.mean(val_acc),
    })
            
    print(f"Val loss: {np.mean(val_loss)}\t Val acc: {np.mean(val_acc)}")
    return np.mean(val_acc)


if __name__ == '__main__':

    cfg = open_config('config.yaml')

    # initialise wandb
    wandb.init(config = cfg, project='assignment1', name = 'code2')
    cfg = wandb.config
    criterion = nn.BCELoss()  

    model = Classification2D(cfg)
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
        
    device = torch.device('cuda:0' if cfg.cuda else 'cpu')

    train_dataset = TrainDatasetReader(file_path = cfg.train_file_path)
    val_dataset = ValDatasetReader(file_path = cfg.val_file_path)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, 
                                batch_size=cfg.batch_size,
                                shuffle=True)

    valloader = torch.utils.data.DataLoader(val_dataset, 
                            batch_size=cfg.val_batch_size,
                            shuffle=True)

    best = 0
    for epoch in range(cfg.epochs):
        train_one_epoch(model, trainloader, optimizer, 
                        criterion, epoch, device, cfg)
        val_acc = eval(model, 
                    valloader, criterion, device, epoch)
        if val_acc > best:
            best = val_acc
            save_model(cfg, model)
            wandb.log({'accuracy': best})
            
    if cfg.do_predict:
        if not cfg.use_trained:
            model = load_model(cfg, model)
            model.eval()
        # using train_dataset
        get_plot(model, trainloader, name = 'q2_done', title= 'q2_done')

    
