import torch
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
