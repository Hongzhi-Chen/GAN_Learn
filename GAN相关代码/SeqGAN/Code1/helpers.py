import torch
from torch.autograd import Variable
from math import ceil

def prepare_generator_batch(samples,start_letter=0,gpu=False):
    batch_size,seq_len = samples.size()
    inp = torch.zeros(batch_size,seq_len)  #输入数据的维度+1
    target = samples #标签就是原本正确的文本
    inp[:,0] = start_letter           #将sample最后一列舍弃，将第一列用0填充
    inp[:,1:] =target[:,:seq_len-1]   #左闭右开

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()
    return inp,target

def prepare_discriminator_data(pos_samples,neg_samples,gpu=False):
    #将正确的样本和错误的样本一起输入（inp）
    inp = torch.cat((pos_samples,neg_samples),0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0]+neg_samples.size()[0]) #表示正负样本的总个数
    target[pos_samples.size()[0]:] = 0 #左闭右开，将负样本标签设为0

    #shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    target = Variable(target)
    inp = Variable(inp)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp,target

def batchwise_sample(gen,num_samples,batch_size):
    #将样本进行分批次处理,以及通过生成器生成样本
    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))  #生成器自己生成样本，但是在我们对生成器进行训练之后
    #按照维数0（行）进行拼接
    return torch.cat(samples,0)[:num_samples]

def batchwise_oracle_nll(gen,oracle,num_samples,batch_size,max_seq_len,start_letter=0,gpu=False):
    #用gen生成的样本在oracle里进行验证
    s = batchwise_sample(gen,num_samples,batch_size)
    #通过生成器生成数据
    oracle_nll = 0
    for i in range(0, num_samples,batch_size):
        inp,target = prepare_generator_batch(s[i:i+batch_size],start_letter,gpu)
        oracle_loss = oracle.batchNLLLoss(inp,target)/max_seq_len
        oracle_nll += oracle_loss.data.item()
    return oracle_nll/(num_samples/batch_size)

