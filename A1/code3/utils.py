from ctypes.wintypes import PINT
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def get_matrix(y_ped, y_tue, name):
    mat = confusion_matrix(y_tue, y_ped)
    df_cm = pd.DataFrame(mat, index = [i for i in ['forest','mountain', 'opencountry', 'street','tallbuilding']],
                  columns = [i for i in ['forest','mountain', 'opencountry', 'street','tallbuilding']])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel('prediction', fontsize=25)
    plt.ylabel('ground truth', fontsize=25)
    plt.savefig(f'{name}.png')


# ADAM 99 best epoch -> CONVERGES
# SDG 113 best epoch --> CONVERGES
# DELTA 198 best epoch --> CONVERGES
