import nets
from Train import trian
if __name__ == '__main__':
    net = nets.Pnet()
    train = trian.TrainNet(net, r'D:\Pyproject\TorchFace\param\pnet.pkl', r'D:\celeba4\12')
    train.train()