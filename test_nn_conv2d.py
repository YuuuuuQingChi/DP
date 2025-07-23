import torchvision
import torchvision.transforms
import torch.utils.data as data
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
# Apply ToTensor() to the images (transform) not the targets (target_transform)
transform = torchvision.transforms.ToTensor()
writer = SummaryWriter("asd")
dataset = torchvision.datasets.CIFAR100(
    "./dataset",
    train=False,
    transform=transform,  # This is for the images
    download=True
)
dataloader = data.DataLoader(dataset, batch_size=64,drop_last=False)

class SJ(nn.Module):
    def __init__(self):
        super(SJ, self).__init__()
        self.cov1 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.cov1(x)
        return x

neural_network = SJ()
step = 0
for data in dataloader:
    imgs, targets = data
    output = neural_network(imgs)
    print(imgs)
    print(output)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("adasdad",output,step)
    step += 1