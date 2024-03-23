import torch
from torch import autograd
import matplotlib.pyplot as plt
from matplotlib import cm


def calc_grad(outputs, inputs):
    '''Calculate gradients for the outputs in respect to the inputs'''
    return autograd.grad(outputs, inputs,
                         grad_outputs=torch.ones_like(outputs),
                         create_graph=True,
                         retain_graph=True)[0]


def surface_plot_vec(x, y, surf_data, title):
    '''Plot a 2D surface plot from vector data x, y, surf_data'''
    fig = plt.figure()
    ax = fig.add_subplot()
    surf = ax.tricontourf(x,y,surf_data, cmap=cm.coolwarm, antialiased=True)
    fig.colorbar(surf)
    plt.title(title)
    plt.show()
