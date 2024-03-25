import torch
import numpy as np
import argparse
from torch.autograd import Variable
from fcn import FCN
from utils import surface_plot_vec
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-e','--exact', type=str, default = "1+6*x**2+x/(1+2*y**2)", help='Change the equation of thermal conductivity (default 1+6*x**2+x/(1+2*y**2))')
opt = parser.parse_args() 


# the exact thermal conductivity could be changed based on the case
h_exact = lambda x, y: eval(opt.exact)

# initiate the FCN
net_arch = [2, 50, 50, 50, 1] # the original architecture of the PINN
tempNet = FCN(net_arch)
tempNet = tempNet.to(device)
condNet = FCN(net_arch)
condNet = condNet.to(device)

tempNet.load_state_dict(torch.load('models/temp_1230.pt'))
condNet.load_state_dict(torch.load('models/cond_1230.pt'))
tempNet.eval()
condNet.eval()

# load the FEM results
data = np.loadtxt('dataset/data_fine_nondim.csv', delimiter=',', skiprows=1)

# create a meshgrid across the domain ((0,1), (0,1))
x_plt=np.arange(0,1,0.02)
y_plt=np.arange(0,1,0.02)
ms_x, ms_y = np.meshgrid(x_plt, y_plt)
# Just because meshgrid is used, we need to do the following adjustm
x_plt = np.ravel(ms_x).reshape(-1,1)
y_plt = np.ravel(ms_y).reshape(-1,1)
# convert the mesh points to tensor variable
inputs = Variable(torch.from_numpy(np.concatenate((x_plt, y_plt), axis=1)).float(),
                                   requires_grad=True).to(device)


# plot the regressed temperature
T_pinn = tempNet(inputs).data.cpu().numpy()
surface_plot_vec(x_plt.flatten(), y_plt.flatten(), T_pinn.flatten(),
                 "Regressed temperature field", "T_regressed")

# plot the regressed thermal conductivity
h_pinn = condNet(inputs).data.cpu().numpy()
surface_plot_vec(x_plt.flatten(), y_plt.flatten(), h_pinn.flatten(),
                 "Regressed thermal conductivity", "h_regressed")

# plot the actual temperature field
surface_plot_vec(data[:,1], data[:,2], data[:,0],
                 "FEM temperature field", "T_exact")

# plot the actual thermal conductivity
h_plt = h_exact(x_plt, y_plt)
surface_plot_vec(x_plt.flatten(), y_plt.flatten(), h_plt.flatten(),
                 "Exact thermal conductivity", "h_exact")

# calculate the relative error on T and h
in_data = Variable(torch.from_numpy(data[:,1:3]).float(),
                                   requires_grad=True).to(device)
T_pinn_data = tempNet(in_data).data.cpu().numpy()
T_actual = data[:,0]
rel_error_T = abs(T_pinn_data - T_actual)/((abs(T_pinn_data) + abs(T_actual))*0.5)
rel_error_h = abs(h_pinn - h_plt)/h_plt


# plot the difference in results
T_diff = T_pinn_data.flatten() - T_actual
h_diff = h_pinn - h_plt
surface_plot_vec(data[:,1], data[:,2], T_diff.flatten(),
                 "Difference plot $T_{pinn}-T_{FEM}$", "T_diff")
surface_plot_vec(x_plt.flatten(), y_plt.flatten(), h_diff.flatten(),
                 "Difference plot $h_{pinn}-h_{exact}$", "h_diff")