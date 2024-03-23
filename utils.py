import torch
from torch import autograd


def calc_grad(outputs, inputs):
    return autograd.grad(outputs, inputs,
                         grad_outputs=torch.ones_like(outputs),
                         create_graph=True,
                         retain_graph=True)[0]