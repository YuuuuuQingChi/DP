import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
train_set = torchvision.datasets.CIFAR10(root= "./dataset",train= True,download=True,transform=torchvision.transforms.Compose ([torchvision.transforms.ToTensor()]))
test_set = torchvision.datasets.CIFAR10(root= "./dataset",train= False,download=True,transform=torchvision.transforms.Compose ([torchvision.transforms.ToTensor()]))
test_loader = DataLoader(dataset=test_set,batch_size=4,shuffle=True,num_workers= 0 ,drop_last=True)

img,target = test_set[1]
print(img.shape)
print(target)
step = 0
writer = SummaryWriter("adadsff")
for data in test_loader:
    imgs , targets = data
    print(img.shape)
    print(targets)
    step+=1
    writer.add_images("test",imgs,step)