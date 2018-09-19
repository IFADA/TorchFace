import os
import torch
import  numpy as np
# a = np.array([[7,8],[4,6]])
# print(a[:,0])
# b = np.array([3,6,3,6])
# print(b.shape)
# from  PIL import  Image,ImageDraw
# img = Image.open(r'D:\celebA\img_celeba\016328.jpg')
# img.show()
# imDraw = ImageDraw.Draw(img)
# imDraw.rectangle((42,85,42+125,85+173),outline='red')
# img.show()
# c = torch.Tensor([[2,4]])
# c.add([[3,4]])
# print(c)
from PIL import  Image

# with open(r'D:\celeba4\12\positive.txt',encoding='UTF-8') as f:
#   s = []
#   s.extend(f.readlines())
#   strs = s[5].strip().split(" ")
 # print(strs)
path = r'D:\celeba4\12'
dataset = []
dataset.extend(open(os.path.join(path,'positive.txt')).readlines())
strs = dataset[0].strip().split(" ")
img_path = os.path.join(path,strs[0])
print(img_path)
img_data = torch.Tensor(np.array(Image.open(img_path)) / 255. - 0.5)
print(dataset[0])
print(img_data)