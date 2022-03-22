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
        for i in range(len(layer_list)):
            self.layers.add_module('fc'+str(i+1),nn.Linear(layer_list[i][0], layer_list[i][1], bias=layer_list[i][2]))
  
    def forward(self, x):
        """
        Forwards the input through the network
        """
        #x = self.relu(self.conv(x))
        #x.register_hook(lambda grad : torch.clamp(grad, min = 0))     #No gradient shall be backpropagated 
                                                                  #conv outside less than 0
      
        # print whether there is any negative grad
        #x.register_hook(lambda grad: print("Gradients less than zero:", bool((grad < 0).any()))) 
        # x=self.fc_last(self.flatten(x))
        #x = nn.relu(self.fc1(x))
        #x = nn.relu(self.fc2(x))
        
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.last_layer(x) 
        #return x

        return self.layers(x)

        
    def update_grad_lastlayer(self, last_grads):
        self.last_grads=last_grads

    def get_grad_lastlayer(self, weight_shape):
        print(f'Weight input grad: {weight_shape}')
        self.last_grads=torch.zeros(weight_shape.shape)

        return self.last_grads


def init_weights(m):
    """
    Weight initialization
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        #m.weight.data.fill_(0.1)
        if m.bias!=None:
            m.bias.data.fill_(0.1)

#input, output, bias
layers=[[2,1,0],[1,2,0]]

net=Net(layers)
#print(net)

net.apply(init_weights)
print(net)

net = net.float()

import numpy as np

np_array=np.array([2,2])
np_array=torch.from_numpy(np_array)
np_array=np_array.float()

target=np.array([0.5,0.5])
target=torch.from_numpy(target)
target=target.float()

gradient_pre=torch.zeros(3)

optimizer = optim.SGD(net.parameters(), lr=0.1)
# zero the gradient buffers

#for name, param in net.named_parameters():
        #print(param.grad)
#        if 'last_layer' in name:
#            param.register_hook(net.get_grad_lastlayer)

for i in range(2):
    out=net(np_array)

    optimizer.zero_grad()


    """
    for name, param in net.named_parameters():
        #print(param.grad)
        if 'last_layer' in name:
            #
            #param.requires_grad=False
            param.register_hook(net.get_grad_lastlayer)
            
            #print(name, param.grad)
            #if param.grad is not None:
                #param.grad*=0
            #print(name, param.grad)
            #param.grad+=1
            #param.grad*=torch.randn(3)#net.get_grad_lastlayer(param.grad)
            
        print(name, param.grad)
    """

    #for name, param in net.named_parameters():
    #    print(name, param.grad)
    out.backward(target)
    print('------------------------------')
    for name, param in net.named_parameters():
        print(name, param.grad)
    optimizer.step()    # Does the update

    print(out)

exit()
net.eval()

test_tens=torch.tensor([1, 1, 1], dtype=torch.float)
print(net(test_tens))



#print(net.named_parameters)

for name, param in net.named_parameters():
    print(param)

# create your optimizer
optimizer = optim.Adam(net.parameters(), lr=0.01)

# zero the gradient buffers
optimizer.zero_grad()
#criterion = nn.MSELoss()
#loss = criterion(output, target)

(1 - out).mean().backward()

loss.backward()
optimizer.step()    # Does the update
#Initialize weights



#Defines a waste loss function
#optimizer=nn.adam()


optimizer.zero_grad()

#Update gradient.
for name, param in net.named_parameters():
    # if the param is from a linear and is a bias
    #weight_shape=param.shape
    #print(f'Test: before {param.grad}')
    #param.grad=torch.randn(weight_shape)
    #print(f'Test: after {param.grad}')

    #param.grad=torch.zeros(grad.shape)
    #print(name, param.grad)
    if "fc_last" in name:
        param.grad=net.get_grad_lastlayer(param.shape)
        #param.register_hook(lambda grad: torch.zeros(grad.shape))
    
    print(param.grad)




print(f'Test: {torch.randn(3)}')
out = net(torch.randn(3))
print(f'out: {out}')

#dummy loss function
(1 - out).mean().backward()

print("The biases are", net.fc1.bias.grad)     #bias grads are zero


for name, param in net.named_parameters():
    # if the param is from a linear and is a bias
    print(name, param.grad)
    if "fc" in name and "bias" in name:
        param.register_hook(lambda grad: torch.zeros(grad.shape))