# coding:utf-8
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, V, dropout):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding = nn.Embedding(V, embedding_size)
        self.lstm = nn.LSTM(self.embedding_size,
                            self.hidden_size,
                            bidirectional=True,
                            batch_first=True)

    # def init_hiiden(self):fixme !!!!!!!
    #     (ho, co) =

    def forward(self, input):
        '''
        input:  batch*max_len
        output:  batch*maxlen_hidden_size we can get h_last by call output[,-1,]

        :param input:
        :return:
        '''
        input_embed = self.embedding(input) # input batch_maxlen
        output, (h_last, c_last) = self.lstm(input_embed)  # input_embed batch

        return output, (h_last, c_last)

class Decoder(nn.Module):
    def __init__(self, embed_size ,hidden_size, labels_size):
        super(Decoder, self).__init__()

        self.input_size = embed_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, labels_size)

    def forward(self, input, h_c):
        '''
        decoder procedure
        :param input:
        :param h:
        :param c:
        :return:
        '''

        # input = Variable(torch.LongTensor([0] * self.hidden_size))  #batch*embed_size
        # for t in range(self.lens):
        #     output, (h, c) = self.lstm(input, (h,c))
        #     input = y[t+1]

        output, (h, c) = self.lstm(input, h_c)
        label = self.linear(h)
        return output, (h, c), label




