import torch 
import torch.nn as nn

class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,10,2, stride = 2)
        self.relu = nn.ReLU()
        self.flatten = lambda x: x.view(-1)
        self.fc1 = nn.Linear(160,5)
   
  
    def forward(self, x):
        x = self.relu(self.conv(x))
        x.register_hook(lambda grad : torch.clamp(grad, min = 0))     #No gradient shall be backpropagated 
                                                                  #conv outside less than 0
      
        # print whether there is any negative grad
        x.register_hook(lambda grad: print("Gradients less than zero:", bool((grad < 0).any())))  
        return self.fc1(self.flatten(x))


net = myNet()

#Defines a waste loss function
criterion=nn.CrossEntropyLoss()

for name, param in net.named_parameters():
    # if the param is from a linear and is a bias
    weight_shape=param.shape
    print(f'Test: before {param.grad}')
    param.grad=torch.randn(weight_shape)
    print(f'Test: after {param.grad}')
    
    param.grad=torch.zeros(grad.shape)
    print(name, param.grad)
    if "fc" in name and "bias" in name:
        param.register_hook(lambda grad: torch.zeros(grad.shape))





out = net(torch.randn(1,3,8,8))

(1 - out).mean().backward()

print("The biases are", net.fc1.bias.grad)     #bias grads are zero


for name, param in net.named_parameters():
    # if the param is from a linear and is a bias
    print(name, param.grad)
    if "fc" in name and "bias" in name:
        param.register_hook(lambda grad: torch.zeros(grad.shape))