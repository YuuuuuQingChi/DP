import torch
from torch import nn
from torch.nn import MaxPool2d
input = torch.tensor(
    [
        [1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1],
    ]
)
input = torch.reshape(input,[1,5,5])
print(input.shape)
class SJ(nn.Module):
    def __init__(self):
        super(SJ, self).__init__()
        self.maxpool = MaxPool2d(3,ceil_mode=True)
    
    def forward(self,input):
        output = self.maxpool(input)
        return output


test = SJ()
out = test(input)
print(out)