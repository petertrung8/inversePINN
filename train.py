import argparse
import time
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataloader import CSVDataset
from fcn import FCN
from models import heatEqRes, boundaryLoss


def train(train_source, input_col, output_col, source_term, n_bc, neumann_bound,
          neumann_val, dirichlet_bound, dirichlet_val,
          num_epoch, batch_size, learning_rate, decay_rate, weight, 
          net_arch, device, save_mod):
    
    # load the dataset into a dataloader
    train_dataset = CSVDataset(train_source, input_col, output_col, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # create boundary data loss
    BCLoss = boundaryLoss(neumann_bound, dirichlet_bound, n_bc,
                          neumann_val, dirichlet_val)
    
    # source term
    q = source_term
    
    # initiate the FCN
    tempNet = FCN(net_arch)
    tempNet = tempNet.to(device)
    condNet = FCN(net_arch)
    condNet = condNet.to(device)
    
    # set the loss function, ooptimizer and the learning rate scheduler
    mse_cost_function = nn.MSELoss() # Mean squared error
    optimizer = optim.Adam([
        {'params': tempNet.parameters()},
        {'params': condNet.parameters()}
        ], lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.97)
    decay_step=decay_rate
    
    for epoch in range(num_epoch):
        for xy_b, T_b in train_loader:
            optimizer.zero_grad() # to make the gradients zero
            x_b = xy_b[:,0].unsqueeze(1)
            y_b = xy_b[:,1].unsqueeze(1)
            # Loss based on BC
            mse_bc = BCLoss(tempNet, mse_cost_function)
            
            # Loss based on data
            T_nn = tempNet(torch.cat([x_b,y_b], axis=1))
            mse_d = mse_cost_function(T_nn, T_b)
            
            # Loss based on PDE
            f_out = heatEqRes(x_b,y_b,q,tempNet,condNet) # output of f(x,t)
            mse_f = mse_cost_function(f_out, torch.zeros_like(T_b))
            
            # Combining the loss functions
            loss = weight[0]*mse_f+weight[1]*mse_d+weight[2]*mse_bc
            loss.backward() 
            optimizer.step()
        if epoch%decay_step == 0 and epoch != 0:
            scheduler.step()
    
        if epoch%100 == 0:
            with torch.autograd.no_grad():
            	print(epoch,"Total Loss:",loss.data.item(),", pde loss:",mse_f.item(),", data loss:", mse_d.item(),\
                      ", bc loss:", mse_bc.item(), ", LR:", scheduler.get_last_lr()[0])
        
    if save_mod:
        date = datetime.now()
        torch.save(tempNet, f'tempFieldNet_{date.strftime('%Y%m%d_%H%M')}.pt')
        torch.save(condNet, f'condFieldNet_{date.strftime('%Y%m%d_%H%M')}.pt')
    
        
    return tempNet, condNet