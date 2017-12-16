# coding:utf-8
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, V):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # print("V = {0}".format(V))
        self.embedding = nn.Embedding(V, input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)  # 随机的按照最好的随机化
        self.linear1.weight = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.linear2.weight = nn.Parameter(torch.zeros(output_size, hidden_size))
        self.linear1.bias = nn.Parameter(torch.zeros(1, hidden_size))
        self.linear2.bias = nn.Parameter(torch.zeros(1, output_size))

    def forward(self, input):
        '''
        forward the procedure
        :param input:
        :return:
        '''
        # print("input = {0}".format(input))
        # print("input size {0}".format(input.size()))
        emb_input = self.embedding(input) # 1 * input_size
        hidden = self.linear1(emb_input)
        # hidden = F.tanh(hidden)
        hidden = F.sigmoid(hidden)
        output = self.linear2(hidden)  #
        out = F.log_softmax(output)  # 对第一个维度做softmax

        return out