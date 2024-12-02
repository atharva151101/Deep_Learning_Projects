from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torch
from torchvision import transforms

def get_matrix(y_ped, y_tue, name):
    mat = confusion_matrix(y_tue, y_ped)
    
    df_cm = pd.DataFrame(mat, index = [i for i in ['cavallo','elefante', 'farfalla', 'gatto','ragno']],
                  columns = [i for i in ['cavallo','elefante', 'farfalla', 'gatto','ragno']])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel('prediction', fontsize=20)
    plt.ylabel('ground truth', fontsize=20)
    plt.savefig(f'{name}.png')

def get_predictions(model, valloader, name, folder):
    y_tue =[]
    y_ped = []
    imgs = []
    device = torch.device('cpu')
    model.to(device)
    with torch.no_grad():
        for data, target in valloader:
            y_tue.extend(target.numpy().tolist())
            data = data.to(device)
            output = model(data)
            output = torch.argmax(output, dim = -1)
            y_ped.extend(output.cpu().numpy().tolist())
            imgs.extend([i for i in data.cpu()])
    
    get_matrix(y_ped, y_tue, name = name)
       


