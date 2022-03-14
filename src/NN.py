import torch
import torch.nn as nn
import torch.nn.functional as F

class Feedforward(nn.Module):
    """
    Neural network class to use in context of Hamiltonian parameters
    """
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

def weights_init_xavier(m):
    """
    Weight function to initialize the weights according to the xavier rule
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        #if m.bias.data is not None:
            #nn.init.xavier_uniform_(m.bias.data)
