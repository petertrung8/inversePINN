import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable


class CSVDataset(Dataset):
    '''CSV data dataloader for fully connected neural network.
    The class is initiated with the CSV data path, list of columns
    corresponding to inputs and list of columns corresponding to outputs.
    Optional header input to skip rows'''
    def __init__(self, csv_file, input_col, output_col,
                 header=1, device='cuda'):
        self.data = np.loadtxt(csv_file, delimiter=',', skiprows=header)
        self.input_data = Variable(torch.from_numpy(self.data[:, input_col]).float(),
                                   requires_grad=True).to(device)
        self.output_data = Variable(torch.from_numpy(self.data[:, output_col]).float(),
                                    requires_grad=True).to(device)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.input_data[index,:], self.output_data[index,:]
