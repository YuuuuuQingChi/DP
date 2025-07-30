from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
class SJ(nn.Module):
    def __init__(self):
        super(SJ, self).__init__()
        # self.conv2d_1 = nn.Conv2d(3,32,kernel_size=5,stride=1,padding=2)
        # self.maxpool_1 = nn.MaxPool2d(2,stride=1)
        # self.conv2d_2 = nn.Conv2d(32,32,5,padding=2)
        # self.maxpool_2 = nn.MaxPool2d(2)
        # self.conv2d_3 = nn.Conv2d(32,64,5,padding=2)
        # self.maxpool_3 = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.linear_1 = nn.Linear(1024,64)
        # self.linear_2 = nn.Linear(64,10)
        nn.L1Loss()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
            
        )

    def forward(self,x):
        return self.sequential(x)
    
Test = SJ()
Test = Test.to('cuda')
visual = SummaryWriter("test")
input = torch.randn((64,3,32,32))

input = input.to('cuda')

output = Test(input)
visual.add_graph(Test,input)
visual.close()