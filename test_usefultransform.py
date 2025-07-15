from PIL import Image
from  torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer =  SummaryWriter("wsws")
img  = Image.open("hymenoptera_data/train/ants/5650366_e22b7e1065.jpg")
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("adad",img_tensor)
# print(img_tensor)

trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
# print(img_norm)
writer.add_image("norm",img_norm)

# print(img.size)
trans_resize = transforms.Resize((512,512))
resize_image = trans_resize(img)
# print(resize_image)
tensor_resize_image = trans_totensor(resize_image)
writer.add_image("adadadf",tensor_resize_image)

resize_image = trans_resize(img_tensor)
# print(resize_image)


trans_compose = transforms.Compose([trans_totensor,trans_resize])
img_resize_2 = trans_compose(img)
# print(img_resize_2.size)

trans_randomcrop = transforms.RandomCrop(
    size=(100, 64),
    padding=20,
    pad_if_needed=True,
    fill= 5,
    padding_mode="constant"
)
img_randomcrop = trans_randomcrop(img_tensor)
writer.add_image("RandomCrop Demo", img_randomcrop, global_step=0)  # 需要指定global_step
writer.close()