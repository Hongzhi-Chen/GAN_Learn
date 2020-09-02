import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import torch.nn.init as init

class Generator(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,max_seq_len,gpu=False,oracle_init=False):
        super(Generator,self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.gru = nn.GRU(embedding_dim,hidden_dim)  #单层单向的GRU网络
        self.gru2out = nn.Linear(hidden_dim,vocab_size)
        #初始化oracl网络的的参数在0和1之间，否则会导致方差太小
        if oracle_init:
            for p in self.parameters():
                init.normal(p,0,1)

    def init_hidden(self,batch_size=1):
        h = autograd.Variable(torch.zeros(1,batch_size,self.hidden_dim))  #(num_layers* 1,batch_size,hidden_size)
        if self.gpu:
            return h.cuda()
        else:
            return  h

    def forward(self, inp,hidden):
        emb = self.embeddings(inp)  #batch_size x embedding_dim
        emb = emb.view(1,-1,self.embedding_dim) # 1 x batch_size x embedding_dim
        out,hidden = self.gru(emb,hidden)    # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1,self.hidden_dim)) # batch_size(32) x vocab_size(5000)
        out = F.log_softmax(out,dim=1)            #对每一行进行归一化处理
        return out,hidden

    # 生成一个序列长度为seq_len的样本，用隐藏层的输出作为下一个单词的输入，这是生成器嘛
    def sample(self,num_samples,start_letter=0):
        #返回样本*最大句子长度
        samples = torch.zeros(num_samples,self.max_seq_len).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out,h = self.forward(inp,h)  #out:num_samples x vocab_Size ，根据GRU的网络结构来的，out是归一化的结果
            out = torch.multinomial(torch.exp(out),1) #num_samples *1(sampling from each row)，每一次out只取权重最大的下标
            samples[:,i] = out.view(-1).data

            inp = out.view(-1)
        #返回生成器生成的句子
        return samples          #返回所有的num_samples

    def batchNLLLoss(self,inp,target):
        #返回预测目标序列的损失函数
        #输入inp为批量大小*句子长度，输入目标为批量大小*句子长度
        #inp should be target with <s> (start letter) prepended

        loss_fn = nn.NLLLoss()
        batch_size,seq_len = inp.size()
        inp = inp.permute(1,0)   #seq_len*batch_size
        target = target.permute(1,0) #seq_len*batch_size，每一列是一个seq
        h = self.init_hidden(batch_size)
        loss = 0
        for i in range(seq_len):
            #所有batch_size中第i个元素
            #out为5000个词可能性的比例
            out,h = self.forward(inp[i],h)
            loss += loss_fn(out,target[i])

        return loss   #每一个批次


    def batchPGLoss(self,inp,target,reward):
        #inp和target与上个batchloss中相同,inp应该以开始字符s作为目标
        #reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        batch_size,seq_len = inp.size()
        inp = inp.permute(1,0)
        target = target.permute(1,0)
        h = self.init_hidden(batch_size)  #[1,batch_size,embedding_dim]

        loss = 0

        for i in range (seq_len):
            #seq_len*batch_size
            out,h = self.forward(inp[i],h)   #out:(batch_size,vocab_size)
            #TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                #第j个样本第i个词语
                #target预测在第j个样本第i个词语的序号，然后在out中查找第j个样本是target.data[i][j]的期望，再乘上奖励值即可
                loss += -out[j][target.data[i][j]]*reward[j]  # log(P(y_t|Y_1:Y_{t-1})) * Q，计算每句话也就是某个状态的奖励

        return loss/batch_size
