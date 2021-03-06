import torch
import torch.autograd as autograd
import torch.nn as nn
import pdb

class Discriminator(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,max_seq_len,gpu=False,dropout=0.2):
        super(Discriminator,self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.gru = nn.GRU(embedding_dim,hidden_dim,num_layers=2,bidirectional=True,dropout=dropout)   #双向双层的GRU网络
        self.gru2hidden = nn.Linear(2*2*hidden_dim,hidden_dim)   #因为双层双向所以需要有2*2
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim,1)

    def init_hidden(self,batch_size):
        h = autograd.Variable(torch.zeros(2*2*1,batch_size,self.hidden_dim))  # (num_layers* 1,batch_size,hidden_size)
        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self,input,hidden):
        #input dim                           #batch_size*seq_len
        emb = self.embeddings(input)      #batch_size*seq_len*embedding_dim
        emb = emb.permute(1,0,2)             #seq_len*batch_size*embedding_dim
        _hidden = self.gru(emb,hidden)       #4*batch_size*hidden_dim
        hidden = hidden.permute(1,0,2).contiguous()  #batch_size*4*hidden_dim
        out = self.gru2hidden(hidden.view(-1,4*self.hidden_dim)) #batch_size*(4*hidden_dim)
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)            #batch_Size*1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self,inp):
        #对一批序列进行分类
        #inp：batch_size*seq_len
        #out：batch_size([0,1]score)
        h = self.init_hidden(inp.size()[0])   #batch_size
        out = self.forward(inp,h)
        return out.view(-1)

    def batchBCELoss(self,inp,target):
        #返回鉴别器二分交叉熵损失函数的
        #inp:batch_size*seq_len
        #target:batch_size(binary 1/0)

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp,h)
        return loss_fn(out,target)