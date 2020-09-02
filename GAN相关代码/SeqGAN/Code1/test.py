import torch
import torch.nn as nn
from torch.autograd import Variable
from math import ceil
import torch.autograd as autograd
import re
import string
import itertools
import numpy as np

"""
def prepare_generator_batch(samples, start_letter=0, gpu=False):
    batch_size, seq_len = samples.size()
    inp = torch.zeros(batch_size, seq_len)  # 输入数据的维度+1
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len - 1]
    print(inp)

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(inp).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


m = torch.zeros(2,3)
m[0,2] = 1
m[1,2] = 2
prepare_generator_batch(m)
"""

#nn.GRU网络模型
"""
# 构建网络模型---输入矩阵特征数input_size、输出矩阵特征数hidden_size、层数num_layers
rnn = nn.LSTM(10,20,2)         # (input_size,hidden_size,num_layers)
inputs = torch.randn(5,3,10)   # (seq_len,batch_size,input_size)
h0 = torch.randn(2,3,20)       # (num_layers* 1,batch_size,hidden_size)
c0 = torch.randn(2,3,20)       # (num_layers*1,batch_size,hidden_size)
num_directions=1               # 因为是单向LSTM

output,(hn,cn) = rnn(inputs,(h0,c0))    # (h0,c0)也可以用none来代替，使系统来初始化

print(output.size())  #torch.Size([5, 3, 20])
print(hn.size())      #torch.Size([2, 3, 20])
print(cn.size())      #torch.Size([2, 3, 20])
"""

#词嵌入向量的应用
"""
word_to_idx = {'hello':0,'world':1}
embeds = nn.Embedding(2,5)
hello_idx = torch.LongTensor([word_to_idx['hello']])
print(hello_idx)
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)


#词嵌入向量非自己初始化
# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[0,2,4,5],[4,3,2,9]])
output = embedding(input)
print(output)


#带补全的词嵌入
# example with padding_idx
embedding = nn.Embedding(10, 3, padding_idx=0)  #可以理解为0相当于一个空格，一个句子里面去除零才是句子的长度，也可以用其它的数字表示空格
input = torch.LongTensor([[0,2,0,9]])   #带补全操作
output = embedding(input)
print(output)



#词嵌入向量自己初始化
# FloatTensor containing pretrained weights
weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)
# Get embeddings for index 1
input = torch.LongTensor([1])
output = embedding(input)
print(output)
"""


"""
#torch.multinomial()
weights = torch.Tensor([0,1,100,0])
print(torch.multinomial(weights,2)) #无放回返回尽可能是大的两个权重的下标


weights = torch.Tensor([0, 10, 3, 0]) # create a Tensor of weights
print(torch.multinomial(weights, 4,replacement=True)) #有放回返回尽可能是大的四个权重的下标
"""


"""
#关于nn.batchNLLLOSS()
#假设有四张图片，图片有三分类（M*N表示有m个样本，每个样本有N个分类），第一列为猫，第二列为狗，第三列为猪
input = torch.randn(3,3)
copy_input = input
print(input)
#对每一行进行一次softmax，得到每张图片的概率分布
sm = nn.Softmax(dim=1)
input = sm(input)
print(input)
#然后对input进行log取对数进行计算
input = torch.log(input)
print(input)
#NLLLoss的结果就是把上面的输出与Label对应的那个值拿出来，再去掉负号，再求均值。
target = torch.tensor([0,2,1])
#假设我们现在Target是[0,2,1]（第一张图片是猫，第二张是猪，第三张是狗）。
#第一行取第0个元素，第二行取第2个，第三行取第1个，去掉负号，结果是：[0.4155,1.0945,1.5285]。再求个均值，结果是：
#最终loss为他们三个的均值

#用NLLLoss进行验证
loss = nn.NLLLoss()
target = torch.tensor([0,2,1])
print(loss(input,target))
"""

"""
inp = autograd.Variable(torch.LongTensor([0]*500))
inp[499] = 1
print(inp.size())
"""

#保留连字符的正则操作
"""
str = "D-typed variables, Python; really?!! god's 'its Gr00vy"
print(re.split(r'[\s\?.\,!]+',str))
"""

"""
#embedding和GRU的使用
#torch.nn.Embedding(n_vocablulary,embedding_size)
#传统构建方法
def check_punction(word):
    for idx,c in enumerate(word):
        if c in string.punctuation:
            return idx
    return -1

corpus = ['I am a boy.','How are you?','I am very lucky.']
batch_list = []
word_set = set()
#将语料中的大写转小写；单词和标点符号分开；
for seq in corpus:
    seq_list = []
    seq = seq.lower()
    for word in seq.split(" "):
        check_val = check_punction(word)
        if check_val != -1:
            word_set.add(word[:check_val])
            word_set.add(word[check_val])
            seq_list.append(word[:check_val])
            seq_list.append(word[check_val])
            continue
        word_set.add(word)
        seq_list.append(word)
    batch_list.append(seq_list)
#print(batch_list)
batch_list.sort(key=lambda i:len(i),reverse=True)
#print(batch_list)


#将单词替换为数字符号
word_2_id = {}
word_2_id['EOS'] = 1 #假设每句话的结尾为EOS
word_2_id['PAD'] = 2 #对于句子长度不够的用PAD进行补充
for word in word_set:
    word_2_id[word] = len(word_2_id)+1
print(word_2_id)


#将句子进行填充
batch_list_temp = []
max_seq_len = len(batch_list[0])
for seq in batch_list:
    pad_num = max_seq_len - len(seq)
    for idx,word in enumerate(seq):
        seq[idx] = word_2_id[word]
    seq.append(word_2_id['EOS'])
    while pad_num != 0:
        seq.append(word_2_id['PAD'])
        pad_num -= 1
    batch_list_temp.append(seq)
batch_list = batch_list_temp


#上面batch有3个样例，RNN的每一步要输入每个样例的一个单词，一次输入batch_size个样例，
#所以batch要按list外层是时间步数(即序列长度)，list内层是batch_size排列。即batch的维度应该是：[seq_len,batch_size]
batch_list = np.array(batch_list)
batch_list = batch_list.T


#[seq_len,batch_size,embedding_size]
batch_list = torch.LongTensor(batch_list)
batch_list = Variable(batch_list)
embed = torch.nn.Embedding(14,6)
embed_batch = embed(batch_list)   #
print(embed_batch)

#将embed_batch代入
input_size = 6
hidden_size = 6
n_layers = 1
#将batch中的补充项句子标记
batch_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_batch,[6,6,5]) #[6,6,5]是原来句子的长度
print(batch_packed)
gru = torch.nn.GRU(input_size,hidden_size,n_layers)
output,hidden  = gru(batch_packed,None)   #hidden = [n_layers,batch_size,hidden_size]，最后一个时间步的batch_size
print(output)
print(hidden)
"""

samples = torch.randn(3,5)
print(samples)
batch_size,seq_len = samples.size()
inp = torch.zeros(batch_size,seq_len)  #输入数据的维度+1
target = samples #标签就是原本正确的文本
inp[:,0] = 0           #将第一列用0填充
inp[:,1:] =target[:,:seq_len-1]   #左闭右开
print(inp)

