from PIL import Image,ImageFilter,ImageDraw
import numpy as np

import torch


# a =torch.Tensor([[1,3],[5,6],[7,9]])
# b= torch.Tensor([[6,8],[4,9],[0,6]])
# d= torch.Tensor([[6,8],[4,9],[0,6]])
# c = torch.cat([a,b,d],dim=1)
# print(c.shape)
# print(c)

#_boxes = boxes[(-boxes[:, 4]).argsort()]
# a = np.array([[1, 2, 3, 4, 5], [9, 8, 3, 4, 1],[3,6,7,6,8]])
# print(-a)
# print('a',-a[:,4])
# print((-a[:,4]).argsort())
# print(a[(-a[:,4]).argsort()])
# print('a.shape[0]',a.shape[0])
# # c =a[1:]
# print('d',a[:1])
# print('c',c)
# a = torch.Tensor([2,4,6,7,8])
# z = torch.nonzero(torch.gt(a,4))
# print(z)
# a =np.array([3])
# b = np.array([1,6,7,8,9])
# #index = np.where(iou(a_box, b_boxes, isMin) < thresh)
# index = np.where(b>a)
# print(index)

# a = np.arange(8).reshape(1,1,8)
# print(a)
# b = a[:,0,1]
# print(b)
# img = Image.open(r'D:\Pyproject\TorchFace\000002.jpg')
# img.show()
# # img = img.resize((300,300))
# a = np.array(img)
# print(a.shape)
# img.show()
# obfuscate_face = img.filter(ImageFilter.BLUR)
# obfuscate_face.show()


# b = np.arange(4).reshape(1,1,2,2)
# print('b',b)
# # print(b[:,1])
# # c =b[0][0]
# # print(c[0],c[1])
# a = np.array([[3,4],[2,6],[3,6]])
# b = np.array([[1,5],[8,0],[1,9]])
# c = np.array([[1,5],[8,0],[1,9]])
# d = np.stack([a,b,c])
# print(d)
# print(d.shape)


# square_bbox = bbox.copy()
# if bbox.shape[0] == 0:
# 	return np.array([])
# h = bbox[:, 3] - bbox[:, 1]
# w = bbox[:, 2] - bbox[:, 0]
# max_side = np.maximum(h, w)
# square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
# square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
# square_bbox[:, 2] = square_bbox[:, 0] + max_side
# square_bbox[:, 3] = square_bbox[:, 1] + max_side
img = Image.open(r'D:\Pyproject\TorchFace\000002.jpg')

# img.show()
# print(img.size)
# w ,h = img.size
# side = np.maximum(w,h)
# #72  94 221 306
# draw= ImageDraw.Draw(img)
# draw.rectangle((72,94,72+221,94+306))
# x1 = 72
# y1= 94
# x2 = 291
# y2 =400
# img = img.crop((x1,y1,x2,y2))
# img.show()
# w_,h_ = img.size
#
# max_side=np.maximum(w_,h_)
# img = img.resize((max_side,max_side))
# img.show()
# x1_ = x1+w_/2- max_side/2
# y1_ = y1+w_/2- max_side/2
# x2_ = x1_+max_side
# y2_ = y1_+ max_side
# img = img.crop((x1_,y1_,x2_,y2_))
# print(img.size)
# img.show()
# a = np.array([2,4,5])
# category_mask = torch.lt(a, 6)  # part样本不参与计算
# category = a[category_mask]
# print(category)

# a = torch.Tensor([[8,2],[2,4],[5,6],[7,8]])
# mask = a[:,0]<6
#
# print('mask',mask)
# print(a[mask])

# img = img.resize((48,48))
# img.show()
# img = img.filter(ImageFilter.BLUR)
#
#
# img.show()
# a = torch.Tensor([[8,2],[2,4],[5,6],[7,8]])
#
# print(a.shape[0])
import  visdom
import torch
import random
import numpy as np
vis = visdom.Visdom(env = 'test1')
x = torch.arange(1,30,0.1)
y = torch.sin(x)
vis.line(X=x,Y=y,win='sinx',opts={'title':'y=sin(x)'})

# #
# # #viz.z =env='my_wind'#设置环境窗口的名称是'my_wind',如果不设置名称就在main中
# # tr_loss=list(range(100))
# # vis.line(Y=np.array(tr_loss), opts=dict(showlegend=True))
# #
# # # 化图像
# # #
# vis = visdom.Visdom(env='test2')
# vis.image(torch.randn(3,64,64).numpy(),opts={'title':'64*64'})
# import visdom
# import time
# vis = visdom.Visdom(env='test3')
# for a in range(10000):
#     if a %10==0:
#         vis.line(Y=a %10==0,X=a)
#         print(a)
