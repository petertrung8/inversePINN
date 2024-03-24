import torch
import numpy as np
from utils import calc_grad


def heatEqRes(x, y, q, temp, cond):
    '''Calculate the residuals from the heat equation given 
    the input data x, y, the source term q, the temperature
    field model and the heat conductitivity model'''
    # get the temperature field and thermal conductivity
    T = temp(torch.cat([x,y], axis=1))
    h = cond(torch.cat([x,y], axis=1))
    
    # calculate the first gradient
    Tx = calc_grad(T, x)
    Ty = calc_grad(T, y)
    
    # calculate the second gradient
    hTx = calc_grad(h*Tx, x)
    hTy = calc_grad(h*Ty, y)
    return hTx+hTy+q


def boundarySquare(bound, n_p, domain=((0,1), (0,1))):
    '''Generate and return the coordinates of boundary points, locations given 
    by a list of string bound, top 't', bottom 'b', left 'l' and right 'r'.
    domain is a two pair tuple of the boundary points 
    ((x_min, x_max),(y_min, y_max)), default ((0,1), (0,1))'''
    x = np.zeros((len(bound)*n_p, 1))
    y = np.zeros((len(bound)*n_p, 1))
    for i in range(len(bound)):
        if bound[i] == 't':
            x[n_p*i:n_p*(i+1),:] = np.linspace(domain[0][0],domain[0][1],n_p)[:,np.newaxis]
            y[n_p*i:n_p*(i+1),:] = np.ones((n_p,1))*domain[1][1]
        if bound[i] == 'b':
            x[n_p*i:n_p*(i+1),:] = np.linspace(domain[0][0],domain[0][1],n_p)[:,np.newaxis]
            y[n_p*i:n_p*(i+1),:] = np.ones((n_p,1))*domain[1][0]
        if bound[i] == 'r':
            x[n_p*i:n_p*(i+1),:] = np.ones((n_p,1))*domain[0][1]
            y[n_p*i:n_p*(i+1),:] = np.linspace(domain[1][0],domain[1][1],n_p)[:,np.newaxis]
        if bound[i] == 'l':
            x[n_p*i:n_p*(i+1),:] = np.ones((n_p,1))*domain[0][0]
            y[n_p*i:n_p*(i+1),:] = np.linspace(domain[1][0],domain[1][1],n_p)[:,np.newaxis]
    return x, y
