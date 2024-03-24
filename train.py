import argparse
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
          net_arch, device, save_mod=True):
    
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='training data csv file, required', required=True)
    parser.add_argument('--in_col', nargs='+', type=int, help='specify the input data columns, required', required=True)
    parser.add_argument('--out_col', nargs='+', type=int, help='specify the output data columns, required', required=True)
    parser.add_argument('--source_term', type=float, default=1, help='the source term q value (default 1)')
    parser.add_argument('--n_bc', type=int, default=25, help='number of points per boundary (default 25)')
    parser.add_argument('--neu_bound', nargs='+', default=['l','t','r'] , help='Specify the location of Neumann boundary on a square (default [l, t, r] - left, top, right)')
    parser.add_argument('--neu_val', type=float, default=0, help='Specify the Neumann boundary value, (default 0)')
    parser.add_argument('--dir_bound', nargs='+', default=['b'] , help='Specify the location of Neumann boundary on a square (default [b] - bottom)')
    parser.add_argument('--dir_val', type=float, default=0, help='Specify the Dirichlet boundary value (default 0)')
    parser.add_argument('--iter', type=int, default=1000, help='Number of training iterations (default 1000)')
    parser.add_argument('--batch_size',type=int, default=1024, help='Batch size for training (default 1024)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for Adam algorithm (default 0.001)')
    parser.add_argument('--decay_r', type=int, default=1000, help='Decay rate (iter) for learning rate (default 1000)')
    parser.add_argument('--w_loss',  nargs='+', type=int, default=[1,5,1], help='weights of each loss term (pde, data, bc order) (default [1, 5, 1]')
    parser.add_argument('--net_arch',  nargs='+', type=int, default=[2,50,50,50,1], help='list specifying the architecture of the trained network (default [2,50,50,50,1])')
    parser.add_argument('--device', default='cuda', help='select the device (default cuda)')
    parser.add_argument('--save', action='store_true', help='save results to folder (default False)')
    opt = parser.parse_args()
    
    train(opt.train_data, opt.in_col, opt.out_col, opt.source_term, opt.n_bc,
          opt.neu_bound, opt.neu_val, opt.dir_bound, opt.dir_val, opt.iter,
          opt.batch_size, opt.lr, opt.decay_r, opt.w_loss, opt.net_arch,
          opt.device, opt.save)
