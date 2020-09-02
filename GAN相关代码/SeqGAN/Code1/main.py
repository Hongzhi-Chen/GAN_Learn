from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers

CUDA = True
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
START_LETTER = 0
BATCH_SIZE =32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

oracle_sample_path = './Data/oracle_samples.trc'
oracle_state_dict_path = './Data/oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = './Data/gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './Data/dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'

def train_generator_MLE(gen,gen_opt,oracle,real_data_samples,epochs):
    """
    极大似然预训练生成器
    """
    for epoch in range(epochs):
        print("epoch %d:"%(epoch+1),end=" ")
        sys.stdout.flush()
        total_loss = 0

        for i in range(0,POS_NEG_SAMPLES,BATCH_SIZE):
            inp,target = helpers.prepare_generator_batch(real_data_samples[i:i+BATCH_SIZE],start_letter=START_LETTER,gpu=CUDA)
            #inp表示生成器输入的内容，target是正常的文本内容，通过target来计算inp的loss优化generator
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp,target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()
            #ceil(POS_NEG_SAMPLES / float(BATCH_SIZE))向上取整得到一共是m个batch_size，将m个batch_size分成10份，通过i来看是不是某百分之十的一部分
            if(i/BATCH_SIZE)%ceil(ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        #each loss in a batch is loss per sample，因为批次是按照seq_len来分别计算每个word的损失的，所以最后除上MAX_SEQ_LEN即为每个样本的loss
        total_loss = total_loss/ ceil(POS_NEG_SAMPLES/float(BATCH_SIZE))/ MAX_SEQ_LEN

        #sample from generator and compute oracle NLL，通过生成器自己生成的样本计算损失（计算生成器自动生成样本的能力）
        oracle_loss = helpers.batchwise_oracle_nll(gen,oracle,POS_NEG_SAMPLES,BATCH_SIZE,MAX_SEQ_LEN,start_letter=START_LETTER,gpu=CUDA)

        print("average_train_NLL=%.4f,oracle_sample_NLL=%.4f"%(total_loss,oracle_loss))

def train_generator_PG(gen,gen_opt,oracle,dis,num_batches):
    #适用策略梯度训练生成器，使用来自鉴别器的奖励
    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)  #长度为64的sample
        inp,target = helpers.prepare_generator_batch(s,start_letter=START_LETTER,gpu=CUDA)
        reward = dis.batchClassify(target)  #概率作为奖励值
        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp,target,reward)
        pg_loss.backward()
        gen_opt.step()

    oracle_loss = helpers.batchwise_oracle_nll(gen,oracle,POS_NEG_SAMPLES,BATCH_SIZE,MAX_SEQ_LEN,start_letter=START_LETTER,gpu=CUDA)
    print('oracle_sample_NLL=%.4f'%oracle_loss)

def train_discriminator(discriminator,dis_opt,real_data_samples,generator,oracle,d_steps,epochs):
    #通过鉴别器对真实数据和生成器生成的数据进行训练
    #样本通过d步得到，鉴别器通过epochs次的训练

    #生成一小部分验证集
    pos_val = oracle.sample(100)
    neg_val = generator.sample(100)
    val_inp,val_target = helpers.prepare_discriminator_data(pos_val,neg_val,gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator,POS_NEG_SAMPLES,BATCH_SIZE)
        dis_inp,dis_target = helpers.prepare_discriminator_data(real_data_samples,s,gpu=CUDA)
        for epoch in range(epochs):
            print('d_step %d epoch %d:' %(d_step+1,epoch+1),end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0,2*POS_NEG_SAMPLES,BATCH_SIZE):
                inp,target = dis_inp[i:i+BATCH_SIZE],dis_target[i:i+BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out,target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if(i/BATCH_SIZE) % ceil(ceil(2*POS_NEG_SAMPLES/float(BATCH_SIZE))/10) == 0:
                    print('.',end='')
                    sys.stdout.flush()

            total_acc /= ceil(2*POS_NEG_SAMPLES/float(BATCH_SIZE))
            total_acc /= float(2*POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / 200.))

if __name__ == '__main__':
    oracle = generator.Generator(GEN_EMBEDDING_DIM,GEN_HIDDEN_DIM,VOCAB_SIZE,MAX_SEQ_LEN,gpu=CUDA)
    # 为oracle真实样本网络进行参数赋值
    oracle.load_state_dict(torch.load(oracle_state_dict_path))
    oracle_samples = torch.load(oracle_sample_path).type(torch.LongTensor) #[10000, 20]真实数据


    # 可以通过在生成器构造函数中传递oracle_init=True来生成一个新的oracle样本网络
    # 可以使用helper .batchwise_sample()生成新的oracle示例

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        oracle_samples = oracle_samples.cuda()

    #生成器的极大相似训练
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, 1)
    #train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, MLE_TRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    #预训练鉴别器
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 1, 1)
    #train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 50, 3)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    #对抗训练
    print('\nStarting Adversarial Training...')
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                               start_letter=START_LETTER, gpu=CUDA)
    print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # 训练生成器
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, 1)

        # 训练鉴别器
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, 5, 3)