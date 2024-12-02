from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt



def plotting3D(x, y, z1, z2 = None, name = None, title = None):
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(x, y, z1, color = "green", s = 5, label = 'model-outputs')
    ax.scatter3D(x, y, z2, color = "blue", s = 5, label = 'true-outputs')
    ax.legend()
    plt.title(f"Model outputs v/s Desired outputs on {title} data")
    ax.set_xlabel('x1', fontsize = 10, rotation = 0)
    ax.set_ylabel('x2', fontsize = 10 )
    ax.set_zlabel('y', fontsize = 10, rotation = 60)
    plt.savefig(f'{name}.png')

# def plotting3D_surface(x,y,z,name=None):
#     fig=plt.figure(figsize = (10, 7))
#     ax = plt.axes(projection ="3d")
#     ax.plot_surface(x,y,z, cmap)
