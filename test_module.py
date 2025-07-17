from torch import nn
import torch
class Test(nn.Module): 
    def __init__(self):
        super().__init__()
    
    def forward(self,x):

        return x + 1
    
x = torch.tensor(1.9)
test_nn = Test()
output = test_nn(x)
print(output)
