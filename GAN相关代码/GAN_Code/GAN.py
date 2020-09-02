import torch
from torch import nn,optim,autograd
import numpy as np
import visdom
import random
import matplotlib.pyplot as plt

h_dim = 400
batchsz = 512
viz = visdom.Visdom()

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.net = nn.Sequential(
            #z:[b,2] -> [b,2]
            nn.Linear(2,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,2)
        )

    def forward(self,z):
        output = self.net(z)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,1)
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

#八个高斯混合模型
def data_generator():
    """
    8-gaussian mixture models
    :return:
    """
    scale = 2
    centers = [
        (1,0),
        (-1,0),
        (0,1),
        (0,-1),
        (1. / np.sqrt(2) , 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
    ]
    centers = [(scale*x,scale*y) for x,y in centers]

    while True:
        dataset = []
        for i in range(batchsz):
            point = np.random.randn(2)*0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.414
        yield dataset

def main():
    torch.manual_seed(23)
    np.random.seed(23)
    device = torch.device('cuda')

    data_iter = data_generator()
    x = next(data_iter)
    #[b,2]
    print(x.shape)

    G = Generator().to(device)
    D = Discriminator().to(device)
    optim_G = optim.Adam(G.parameters(),lr=5e-4,betas=(0.5,0.9))
    optim_D = optim.Adam(D.parameters(),lr=5e-4,betas=(0.5,0.9))

    viz.line([[0,0]],[0],win='loss',opts=dict(title='loss',legend=['D','G']))

    print(G)
    print(D)

    for epoch in range(5000):

        #1.train Discrimator firstly
        for _ in range(5):
            #1.1train on real data
            x = next(data_iter)
            x = torch.from_numpy(x).cuda()
            #[b,2] -> [b,1]
            predr = D(x)
            #max predict real data
            lossr = predr.mean()

            #1.2train on fake data
            #[b,2]
            z = torch.randn(batchsz,2).to(device)
            #结束梯度向Generator传播
            xf = G(z).detach() #tf.stop_gradient()
            predf = D(xf)
            lossf = predr.mean()

            #aggregate all
            loss_D = lossr + lossf

            #optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        #2.train Generator
        z = torch.randn(batchsz,2).to(device)
        xf = G(z)
        predf = D(xf)
        #max predict fake data
        loss_G = -predf.mean()

        #optimize
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()


        if epoch%100 == 0:
            viz.line([[loss_D.item(),loss_G.item()]],[epoch],win='loss',update='append')
            print(loss_D.item(),loss_G.item())




if __name__ == '__main__':
    main()

