from torch.utils.data import Dataset
from PIL import Image
import os
class DataTest(Dataset):
    
    def __init__(self,data_root,data_label):
        self.data_root = data_root
        self.data_label = data_label
        self.path = os.path.join(self.data_root,self.data_label)
        self.image_path = os.listdir(self.path)
        
    def __getitem__(self, index):
        image_name = self.image_path[index]
        image_path = os.path.join(self.data_root,self.data_label,image_name)
        image = Image.open(image_path)
        label = self.data_label
        return image, label
    
    def __len__(self):
        return len(self.image_path)

ants_dataset = DataTest("hymenoptera_data/train","ants")
bees_dataset = DataTest("hymenoptera_data/train","bees")
traindataset = ants_dataset + bees_dataset
image, label = traindataset[123]
image.show()