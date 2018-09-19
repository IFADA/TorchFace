import nets
from Train import train
if __name__ == '__main__':
    net = nets.Onet()
    train = train.TrainNet(net,r'C:\celeba4\48')