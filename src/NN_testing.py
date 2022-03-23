import torch 
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    """
    Class: Neural network
    """
    def __init__(self, layer_list):
        super().__init__()
        self.layers = nn.Sequential()
        #Adds linear layers according to the input list
        for i in range(len(layer_list)):
            self.layers.add_module('fc'+str(i+1),nn.Linear(layer_list[i][0], layer_list[i][1], bias=layer_list[i][2]))
  
    def forward(self, x):
        """
        Forwards the input through the network
        """
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
            model.bias.data.fill_(0.1)

#input, output, bias
layers=[[2,1,0],[1,2,0]]

net=Net(layers)
#print(net)

#Initialize weights
net.apply(init_weights)
#Floating the network parameters
net = net.float()

"""
np_array=np.array([2,2])
np_array=torch.from_numpy(np_array)
np_array=np_array.float()
"""

gradient_pre=torch.zeros(3)

optimizer = optim.SGD(net.parameters(), lr=0.1)

for i in range(2):
    out=net(np_array)

    optimizer.zero_grad()

    out.backward(target)
    print('------------------------------')
    for name, param in net.named_parameters():
        print(name, param.grad)
    optimizer.step()    # Does the update

    print(out)
