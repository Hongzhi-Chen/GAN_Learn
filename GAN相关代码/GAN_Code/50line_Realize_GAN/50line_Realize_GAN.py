# Generative Adversarial Networks (GAN) example in PyTorch. Tested with PyTorch 0.4.1, Python 3.6.7 (Nov 2018)

# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9
#正太分布曲线的拟合
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable

matplotlib_is_available = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data params
data_mean = 4   #期望
data_stddev = 1.25   #方差
# ### Uncomment only one of these to define what data is actually sent to the Discriminator
#(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
#(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)
#(name, preprocess, d_input_func) = ("Data and diffs", lambda data: decorate_with_diffs(data, 1.0), lambda x: x * 2)
(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)
print("Using data [%s]" % (name))

# ##### DATA: Target data and generator input data

#得到正确的数据，用来训练判别器D
def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n))).to(device)  #Gaussian

#数据用来喂generator生成器，相当于输入的噪音
def get_generator_input_sampler():
    return lambda m, n: (torch.rand(m, n)).to(device)  # Uniform-dist data into generator, _NOT_ Gaussian

# ##### MODELS: Generator model and discriminator model
#生成器模为全连接层
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x

#鉴别器模型
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))

def extract(v):

    return v.data.storage().tolist()

def stats(d):
    #返回均值和方差
    return ([np.mean(d), np.std(d)])

def get_moments(d):
    # Return the first 4 moments of the data provided
    #得出各种相关的参数
    mean = torch.mean(d)
    diffs = d - mean      #得出每个数据和均值的差值
    var = torch.mean(torch.pow(diffs, 2.0))       #计算方差
    std = torch.pow(var, 0.5)      #计算标准差
    zscores = diffs / std       #标准分数
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
    final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
    return final

def decorate_with_diffs(data, exponent, remove_raw_data=False):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    if remove_raw_data:
        return torch.cat([diffs], 1)
    else:
        return torch.cat([data, diffs], 1)

def train():
    # Model parameters
    g_input_size = 1      # Random noise dimension coming into generator, per output vector
    g_hidden_size = 5     # Generator complexity
    g_output_size = 1     # Size of generated output vector
    d_input_size = 500    # Minibatch size - cardinality of distributions
    d_hidden_size = 10    # Discriminator complexity
    d_output_size = 1     # Single dimension for 'real' vs. 'fake' classification
    minibatch_size = d_input_size
    d_learning_rate = 1e-3
    g_learning_rate = 1e-3
    sgd_momentum = 0.9  #sgd动量优化
    num_epochs = 1000   #总的迭代次数
    print_interval = 100   #间隔多少次进行打印
    d_steps = 20        #每次迭代的训练次数
    g_steps = 20
    dfe, dre, ge = 0, 0, 0
    d_real_data, d_fake_data, g_fake_data = None, None, None
    discriminator_activation_function = torch.sigmoid    #激活函数
    generator_activation_function = torch.tanh
    d_sampler = get_distribution_sampler(data_mean, data_stddev)
    gi_sampler = get_generator_input_sampler()
    G = Generator(input_size=g_input_size,
                  hidden_size=g_hidden_size,
                  output_size=g_output_size,
                  f=generator_activation_function)
    D = Discriminator(input_size=d_input_func(d_input_size),
                      hidden_size=d_hidden_size,
                      output_size=d_output_size,
                      f=discriminator_activation_function)
    G.to(device)
    D.to(device)
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)   #优化器
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)
    for epoch in range(num_epochs):
        #训练鉴别器
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()     #将参数值清空
            #  1A: Train D on real
            #真实数据集的训练
            d_real_data = Variable(d_sampler(d_input_size))
            d_real_decision = D(preprocess(d_real_data))
            d_real_error = criterion(d_real_decision.unsqueeze(0), Variable(torch.ones([1,1]).to(device))).to(device)  # ones = true
            d_real_error.backward() # compute/store gradients, but don't change params
            #  1B: Train D on fake
            #生成器生成数据集的训练
            d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size).to(device))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(preprocess(d_fake_data.t()))
            d_fake_error = criterion(d_fake_decision.unsqueeze(0), Variable(torch.zeros([1,1]).to(device))).to(device)  # zeros = fake
            d_fake_error.backward()
            #真假数据在一起进行优化
            d_optimizer.step()   # Only optimizes D's parameters; changes based on stored gradients from backward()
            dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

        #训练生成器
        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()
            gen_input = Variable(gi_sampler(minibatch_size, g_input_size).to(device))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(preprocess(g_fake_data.t()))    #将错误的结果给到鉴别器D，但并不反向传播进行参数的优化
            g_error = criterion(dg_fake_decision.unsqueeze(0), Variable(torch.ones([1,1]).to(device))).to(device)  # Train G to pretend it's genuine
            #我们希望鉴别器得到的结果是1，所以我们反过来调整gen_input
            g_error.backward()
            g_optimizer.step() # Only optimizes G's parameters
            ge = extract(g_error)[0]

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
                  (epoch, dre, dfe, ge, stats(extract(d_real_data)), stats(extract(d_fake_data))))

    if matplotlib_is_available:
        print("Plotting the generated distribution...")
        values = extract(g_fake_data)
        print(" Values: %s" % (str(values)))
        plt.hist(values, bins=50)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram of Generated Distribution')
        plt.grid(True)
        plt.show()
train()
