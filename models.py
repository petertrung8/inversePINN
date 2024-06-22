import torch
import numpy as np
from torch.autograd import Variable
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


class boundaryLoss():
    def __init__(self, neumann_bound, dirichlet_bound, n_bc,
                 neu_val=0, dir_val=0):
        if neumann_bound:
            self.n_bound = neumann_bound
            self.NeuBC = True
            self.x_Neu, self.y_Neu = self.boundarySquare(self.n_bound, n_bc)
            self.neu_Value = neu_val
        if dirichlet_bound:
            self.DirBC = True
            self.x_Dir, self.y_Dir = self.boundarySquare(dirichlet_bound, n_bc)
            self.dir_Value = dir_val

    def boundarySquare(self, bound, n_p, domain=((0,1), (0,1)), device='cuda'):
        '''Generate and return the coordinates of boundary points, locations given 
        by a list of string bound, top 't', bottom 'b', left 'l' and right 'r'.
        domain is a two pair tuple of the boundary points 
        ((x_min, x_max),(y_min, y_max)), default ((0,1), (0,1))'''
        # initialize the arraying
        x = np.zeros((len(bound)*n_p, 1))
        y = np.zeros((len(bound)*n_p, 1))
        
        # Evaluate the locations at the squares
        for i in range(len(bound)):
            if bound[i] == 't':
                x[n_p*i:n_p*(i+1),:] = np.linspace(domain[0][0],
                                                   domain[0][1],n_p)[:,np.newaxis]
                y[n_p*i:n_p*(i+1),:] = np.ones((n_p,1))*domain[1][1]
            if bound[i] == 'b':
                x[n_p*i:n_p*(i+1),:] = np.linspace(domain[0][0],
                                                   domain[0][1],n_p)[:,np.newaxis]
                y[n_p*i:n_p*(i+1),:] = np.ones((n_p,1))*domain[1][0]
            if bound[i] == 'r':
                x[n_p*i:n_p*(i+1),:] = np.ones((n_p,1))*domain[0][1]
                y[n_p*i:n_p*(i+1),:] = np.linspace(domain[1][0],
                                                   domain[1][1],n_p)[:,np.newaxis]
            if bound[i] == 'l':
                x[n_p*i:n_p*(i+1),:] = np.ones((n_p,1))*domain[0][0]
                y[n_p*i:n_p*(i+1),:] = np.linspace(domain[1][0],
                                                   domain[1][1],n_p)[:,np.newaxis]
        
        # Convert the boundary points to torch variables
        x_t = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        y_t = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
        return x_t, y_t
    
    
    def dirichletLoss(self, x, y, bc_val, model):
        T_d = model(torch.cat([x,y], axis=1))
        diric = torch.ones_like(T_d)*bc_val
        return T_d, diric
    
    
    def neumannLoss(self, x, y, bc_val, model):
        idx_y = torch.zeros_like(y)
        idx_x = torch.zeros_like(x)
        if ('t' in self.n_bound) or ('b' in self.n_bound):
            idx_y = torch.where(y == 1, 1.0, 0.0) + torch.where(y == 0, 1.0, 0.0)
        if ('r' in self.n_bound) or ('l' in self.n_bound):
            idx_x = torch.where(x == 1, 1.0, 0.0) + torch.where(x == 0, 1.0, 0.0)
        idx_x = idx_x > 0
        idx_y = idx_y > 0
        x_x = x[idx_x]
        y_x = y[idx_x]
        x_y = x[idx_y]
        y_y = y[idx_y]
        T_n_x = model(torch.cat([x_x,y_x], axis=1))
        T_n_y = model(torch.cat([x_y,y_y], axis=1))
        dT_n =torch.cat([calc_grad(T_n_x, x_x), calc_grad(T_n_y, y_y)])
        neum = torch.ones_like(dT_n)*bc_val
        return dT_n, neum

    def __call__(self, model, loss_f):
        T_d, diric = self.dirichletLoss(self.x_Dir, self.y_Dir,
                                        self.dir_Value, model)
        dT_n, neum = self.neumannLoss(self.x_Neu, self.y_Neu,
                                      self.neu_Value, model)
        return loss_f(torch.cat([T_d, dT_n], axis=0),
                      torch.cat([diric, neum], axis=0))
