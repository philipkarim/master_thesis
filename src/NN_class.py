import torch 
import torch.nn as nn
import sys
torch.manual_seed(1324)


class Net(nn.Module):
    """
    Class: Neural network
    """
    def __init__(self, layer_list, sample, output_size):
        super().__init__()
        self.layers = nn.Sequential()
        activation_indices=[]

        #Tracking the indices of the linear layers
        for i, j in enumerate(layer_list):
            if not isinstance(j[0], str):
               activation_indices.append(i)

        try:
            self.layers.add_module('fc'+str(0),nn.Linear(len(sample), layer_list[0][0], bias=layer_list[0][1]))
        except:
            self.layers.add_module('fc'+str(0),nn.Linear(len(sample), layer_list[1][0], bias=layer_list[1][1]))

        #Adds linear layers according to the inserted list
        for i, layer in enumerate(layer_list):
            if layer[0]=='relu':
                self.layers.add_module('relu'+str(i+1),nn.ReLU())
            elif layer[0]=='sigmoid':
                self.layers.add_module('sigmoid'+str(i+1),nn.Sigmoid())
            elif layer[0]=='elu':
                self.layers.add_module('elu'+str(i+1),nn.ELU())
            elif layer[0]=='leakyrelu':
                self.layers.add_module('leakyrelu'+str(i+1),nn.LeakyReLU())
            else:
                try:
                    index_layer=activation_indices[activation_indices.index(i)+1]
                    self.layers.add_module('fc'+str(i+1),nn.Linear(layer[0], layer_list[index_layer][0], bias=layer[1]))
                except:
                    #Output layer
                    self.layers.add_module('fc'+str(i+1),nn.Linear(layer[0], output_size, bias=0))

        
    def forward(self, x):
        """
        Forwards the input through the network
        
        Args:
                x(array): Assumes the input are numpy arrays

        Return:
                Input x forwarded through the network
        """
        x=torch.tensor(x).float()

        return self.layers(x)
    

def init_weights(model, initialization='xavier_normal'):
    """
    Weight initialization, 
    """
    if isinstance(model, nn.Linear):
        if initialization=='xavier_normal':
            torch.nn.init.xavier_normal_(model.weight)

        elif initialization=='xavier_uniform':
            torch.nn.init.xavier_uniform_(model.weight)

        elif initialization=='he_normal':
            torch.nn.init.kaiming_normal_(model.weight)

        elif initialization=='he_uniform':
            torch.nn.init.kaiming_uniform_(model.weight)

        elif initialization=='normal':
            torch.nn.init.normal_(model.weight)

        elif initialization=='uniform':
            torch.nn.init.uniform_(model.weight)

        else:
            sys.exit('Neural network nitialization not known')

        #m.weight.data.fill_(0.1)
        if model.bias!=None:
            model.bias.data.fill_(0.01)
