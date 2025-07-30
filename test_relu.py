import torch
from torch import nn
from torch.nn import ReLU
input = torch.tensor([[1, -0.5], [-1, 3]])
print(input.shape)
class SJ(nn.Module):
    def __init__(self):
        super(SJ, self).__init__()
        self.relu1 = ReLU()

    def forward(self, x):
        x = self.relu1(x)
        return x

Test = SJ()
output = Test(input)
print(output.shape)
print(output)