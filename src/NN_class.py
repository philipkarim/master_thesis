import torch 
import torch.nn as nn

class Net(nn.Module):
    """
    Class: Neural network
    """
    def __init__(self, layer_list, sample, output_size):
        super().__init__()
        self.layers = nn.Sequential()
        
        index=0
        #Adds linear layers according to the input list
        for i in range(len(layer_list)-1):
            if layer_list[i][0]=='relu':
                self.layers.add_module('relu+'+str(i+1),nn.ReLU())
            elif layer_list[i][0]=='sigmoid':
                self.layers.add_module('sigmoid+'+str(i+1),nn.Sigmoid())
            elif layer_list[i][0]=='elu':
                self.layers.add_module('elu+'+str(i+1),nn.ELU())
            elif layer_list[i][0]=='leakyrelu':
                self.layers.add_module('leakyrelu+'+str(i+1),nn.LeakyReLU())
            else:
                self.layers.add_module('fc'+str(i+1),nn.Linear(layer_list[i][0], layer_list[i+1][0], bias=layer_list[i][1]))

        #Add output layer
        self.layers.add_module('fc'+str(len(layer_list)),nn.Linear(layer_list[-1][0], output_size, bias=0))
        
    def forward(self, x):
        """
        Forwards the input through the network
        
        Args:
                x(array): Assumes the input are numpy arrays

        Return:
                Input x forwarded through the network
        """

        x=torch.from_numpy(x).float()

        return self.layers(x)
    
    #TODO: Might want to remove these
    def update_grad_lastlayer(self, last_grads):
        self.last_grads=last_grads

    def get_grad_lastlayer(self, weight_shape):
        print(f'Weight input grad: {weight_shape}')
        self.last_grads=torch.zeros(weight_shape.shape)

        return self.last_grads


def init_weights(model):
    """
    Weight initialization, xavier_uniform
    """
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
        #m.weight.data.fill_(0.1)
        if model.bias!=None:
            model.bias.data.fill_(0.01)
