import os
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.optim as optim
from DataFaceset import FaceDataset
from torch import nn
import nets


class TrainNet:
    def __init__(self, net, save_path, dataset_path, isCuda=True):
        self.net = net
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.isCuda = isCuda

        if isCuda:
            self.net.cuda()
        # 定义损失
        self.category_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()
        # 定义优化器
        self.optimizer = optim.Adam(self.net.parameters())
        # 存储器
        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))

    def train(self):
        # 数据集
        faceDateset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(faceDateset, batch_size=10, shuffle=True, num_workers=4, drop_last=True)

        while True:
            for i, (_img_data, _category, _offset) in enumerate(dataloader):
                if self.isCuda:
                    img_data_ = _img_data.cuda()
                    category_ = _category.cuda()
                    offset_ = _offset.cuda()

                    _output_category, _output_offset = self.net(img_data_)
                    output_category = _output_category.view(-1, 1)
                    output_offset = _output_offset.view(-1, 4)

                    # 计算分类的损失
                    print('Debug:category',category_)
                    category_mask = torch.lt(category_, 2)  # part样本不参与分类损失计算
                    category = category_[category_mask]
                    output_category = output_category[category_mask]
                    category_loss= self.category_loss_fn(output_category, category)

                    # 计算bound的损失
                    offset_mask = torch.gt(category_, 0)  # 负样本不参与计算
                    print('Debug:offset_mask',offset_mask)
                    offset = offset_[offset_mask[:,0]]
                    print('Debug:offset',offset)
                    print('Debug:offset_mask[:,0]',offset_mask[:,0])
                    output_offset = output_offset[offset_mask[:,0]]
                    print('Debug:output_offset',output_offset)
                    offset_loss = self.offset_loss_fn(output_offset, offset)  # 损失
                    loss = category_loss + offset_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    import visdom

                    Totalloss = loss.cpu().data.numpy()
                    #vis = visdom.Visdom(env='faceLoss')
                    # for i in range(1000000):
                    #    if i %10==0:
                    #     vis.line(X=Totalloss,Y=i, opts={'title':'faceLoss'})
                    print('loss',Totalloss, '置信度损失', category_loss.cpu().data.numpy(), '偏移量损失',
                              offset_loss.cpu().data.numpy())


            # torch.save(self.net.state_dict(), self.save_path)
            # print('save success')


if __name__ == '__main__':
      net = nets.Onet()
      train = TrainNet(net, r'D:\Pyproject\TorchFace\param\onet.pkl', r'D:\celeba4\48')
      train.train()
