from torchvision import transforms
import numpy
from PIL import Image
import cv2

image_path = "hymenoptera_data/train/bees/16838648_415acd9e3f.jpg"
# img = Image.open(image_path)
img_cv = cv2.imread(image_path)
img = numpy.array(img_cv)
tensor_trans = transforms.ToTensor()
image_tensor = tensor_trans(img)
print(image_tensor)
