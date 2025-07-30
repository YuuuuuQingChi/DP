from torch.utils.data import DataLoader
import torchvision
import torch 
from torch import nn
from torch.nn import Linear
dataset_ = torchvision.datasets.CIFAR10("/data",train=False,transform=  torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset_,batch_size=64,drop_last=True)
class SJ(nn.Module):
    def __init__(self):
        super(SJ, self).__init__()
        self.linear_ = Linear(196608,10)
        
    def forward(self,x):
        x = self.linear_(x)
        return x    

tudui = SJ()


for data in dataloader:
    imgs,targets = data
    # print(imgs.shape)
    output = torch.reshape(imgs,[1,1,1,-1])

    test = torch.flatten(imgs)
    # if(test.shape == output.shape):
    #     print("dad")
    print(test.shape)
    print(output.shape)

    output_ = tudui(output)
    # print(output_.shape)