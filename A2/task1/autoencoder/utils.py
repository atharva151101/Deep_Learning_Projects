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
    #count = 0
    #for i,j,img in zip(y_tue, y_ped, imgs):
    #    k = {'cavallo': 0, 'elefante': 1, 'farfalla': 2, 'gatto': 3, 'ragno': 4}
    #    inv_map = {
    #        jj:ii for ii,jj in k.items()
    #    }
    #    if i!=j:
    #        count +=1
    #        fig, ax = plt.subplots()
            # mean : (array([0.5276365, 0.508226 , 0.4184626], dtype=float32),
            # std : array([0.27150184, 0.26589277, 0.28562558], dtype=float32))
    #        inv_normalize = transforms.Normalize(
    #            mean=[-0.5276365/0.27150184, -0.508226/0.26589277, -0.4184626/0.28562558],
    #            std=[1/0.27150184, 1/0.26589277, 1/0.28562558]
    #        )
    #        inv_tensor = inv_normalize(img)
    #        img = inv_tensor.permute((1,2,0)).numpy().tolist()
    #        ax.imshow(img)
    #        ax.spines['top'].set_visible(False)
    #        ax.spines['left'].set_visible(False)
    #        ax.spines['bottom'].set_visible(False)
    #        ax.spines['right'].set_visible(False)
    #        ax.set_xticks([])
    #        ax.set_yticks([])
    #        plt.savefig(folder + f'{inv_map[i]}_{inv_map[j]}_{count}.png')
       


