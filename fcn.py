import torch
from torch import nn


class FCN(nn.Module):
    # Neural Network
    def __init__(self, layers, activation=nn.Tanh()):
        '''Initialise neural network using a list and an activation function, 
        where each of the elements is a layer and each element is number of
        neurons in the layer.'''
        super().__init__()
        self.layers = layers
        self.activation = activation
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], 
                                                self.layers[i + 1]) for i in 
                                      range(len(self.layers) - 1)])
        # Initialize the weights and biases in the network
        for i in range(len(self.layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)
    
    # Forward pass
    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
