from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(x1, x2, z1 ,xx,yy,zz):
    fig = plt.figure(figsize = (10, 7))
    plt.contourf(xx, yy, zz, colors=('#ffcccb','#90EE90'))
    plt.scatter(x1[z1==1],x2[z1==1],c='green',s=10)
    plt.scatter(x1[z1==0],x2[z1==0],c='red',s=10)
    plt.xlabel("x1",fontsize=10)
    plt.ylabel("x2",fontsize=10)
    plt.title(f"Decision Region")
    plt.savefig('plotr.png')
    
    
