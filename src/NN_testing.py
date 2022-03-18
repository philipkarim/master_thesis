import torch 
import torch.nn as nn

class myNet(nn.Module):
    """
    Class: NEural network
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)  # 5*5 from image dimension
        self.fc2 = nn.Linear(2, 3)
        #self.fc3 = nn.Linear(84, 3)

  
    def forward(self, x):
        #x = self.relu(self.conv(x))
        #x.register_hook(lambda grad : torch.clamp(grad, min = 0))     #No gradient shall be backpropagated 
                                                                  #conv outside less than 0
      
        # print whether there is any negative grad
        #x.register_hook(lambda grad: print("Gradients less than zero:", bool((grad < 0).any()))) 
        # x=self.fc_last(self.flatten(x))
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        #x = self.fc3(x) 
        return x

    def update_grad_lastlayer(self, last_grads):
        self.last_grads=last_grads

    def get_grad_lastlayer(self, weight_shape):
        self.last_grads=torch.randn(weight_shape)

        return self.last_grads


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)


net = myNet()

#Defines a waste loss function
#optimizer=nn.adam()


net.zero_grad()

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