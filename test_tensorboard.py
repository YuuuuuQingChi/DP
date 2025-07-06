from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer = SummaryWriter("test")
image = Image.open("hymenoptera_data/train/ants/5650366_e22b7e1065.jpg")
image_array = np.array(image)
print(image_array.shape)
writer.add_image("1212",image_array,1,dataformats = 'HWC')
print(type(image_array))
writer.close()
