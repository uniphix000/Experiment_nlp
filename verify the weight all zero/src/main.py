# coding: utf-8
import argparse
from dataloader import *
import random
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from module import *

def main():
    cmd = argparse.ArgumentParser("seq2seq for segmentor")
    cmd.add_argument("--train_input_path", help = "path of input ", default='../data/CTB5.1-train.dat')
    cmd.add_argument("--test_input_path", help="path of test ", default='../data/CTB5.1-test.dat')
    cmd.add_argument("--valid_input_path", help="path of valid ", default='../data/CTB5.1-devel.dat')
    cmd.add_argument("--max_epoch", help = "epoch for training", default=10)
    cmd.add_argument("--lr", help = "learning rato for training", default= 0.1)
    cmd.add_argument("--batch_size", help = "batch_size for training", default=4)
    cmd.add_argument("--seed", help = "seed for training", default=1)
    cmd.add_argument("--optimizer",help = "optimizer for training", default='adam')

    args = cmd.parse_args()

    train_x, train_y, valid_x, valid_y, test_x, test_y = data_loader(args.train_input_path, args.test_input_path, args.valid_input_path)
    lang = Lang()
    print("now, generate dict.......")
    lang = generate_dict(lang, test_x)
    train_x_idx, train_y_idx,  valid_x_idx, valid_y_idx, test_x_idx, test_y_idx = word_to_idx(lang, train_x, train_y, valid_x, valid_y, test_x, test_y)
    padding_idx = lang.get_word_to_idx()['pad']
    # V =  len(test_x)
    V = lang.get_word_size()
    print("V = {0}".format(V))
    model = MLP(100, 200, 4, V)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    res = []
    count = 0
    flag  =False
    loss = nn.NLLLoss()
    for t in range(args.max_epoch):
        res = []
        for index_sentence, sentence in enumerate(test_x_idx):
            for index_word, word in enumerate(sentence):
                count += 1
                if count == 500:
                    flag = True
                    break
                model.zero_grad()
                # out = model.forward(Variable(torch.LongTensor(word)))
                out = model.forward(Variable(torch.LongTensor([word]).view(1)))
                # print(out)


                loss_every_data = loss(out.view(1,4), Variable(torch.LongTensor([test_y_idx[index_sentence][index_word]])))
                loss_every_data.backward()
                #print (loss_every_data.data[0])
                res.append(loss_every_data.data[0])
                optimizer.step()
            if flag == True:
                break
               # print(loss_every_data)
        print(model.linear1.weight.grad.sum())
        print(model.linear2.weight.grad.sum())
        print(model.linear1.bias.grad.sum())
        print(model.linear2.bias.grad.sum())

        print("wait")
        print(res)
if __name__=='__main__':
    main()