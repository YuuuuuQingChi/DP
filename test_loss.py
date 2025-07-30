# import torch
# from torch.nn import L1Loss

# input = torch.tensor([1,2,4],dtype=torch.float32)
# output = torch.tensor([2,3,1],dtype=torch.float32)
# input = torch.reshape(input,[1,1,1,-1])
# output = torch.reshape(output,[1,1,1,-1])

# loss = L1Loss(reduction='mean')
# result = loss(input,output)
# print(result)


from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision

datasets_ = datasets.CIFAR10(
    "/dataset", False, transform=torchvision.transforms.ToTensor(), download=False
)
dataloader = DataLoader(datasets_,batch_size=64)
loss = torch.nn.CrossEntropyLoss()
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

        self.sequential = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.sequential(x)


Test = SJ()
for data in dataloader:
    imgs,targets = data
    output = Test(imgs)
    result = loss(output,targets)
    result.backward()
    print(result)
