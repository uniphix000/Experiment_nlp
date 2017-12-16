# coding: utf-8
import argparse
from dataloader import *
import random
from module import *
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

def main():
    cmd = argparse.ArgumentParser("seq2seq for segmentor")
    cmd.add_argument("--train_input_path", help = "path of input ", default='../data/CTB5.1-train.dat')
    cmd.add_argument("--test_input_path", help="path of test ", default='../data/CTB5.1-test.dat')
    cmd.add_argument("--valid_input_path", help="path of valid ", default='../data/CTB5.1-devel.dat')
    cmd.add_argument("--max_epoch", help = "epoch for training", default=1)
    cmd.add_argument("--embedding_size", help="embedding for training", default=100)
    cmd.add_argument("--hidden_size", help="hidden for training", default=128)
    cmd.add_argument("--lr", help = "learning rato for training", default='0.001')
    cmd.add_argument("--batch_size", help = "batch_size for training", default=4)
    cmd.add_argument("--dropout", help="dropout for training", default=0.75)
    cmd.add_argument("--seed", help = "seed for training", default=1)
    cmd.add_argument("--optimizer",help = "optimizer for training", default='adam')

    args = cmd.parse_args()

    train_x, train_y, valid_x, valid_y, test_x, test_y = data_loader(args.train_input_path, args.test_input_path, args.valid_input_path)
    train_x = test_x
    train_y = test_y
    lang = Lang()
    print("now, generate dict.......")
    lang = generate_dict(lang, train_x)
    train_x_idx, train_y_idx,  valid_x_idx, valid_y_idx, test_x_idx, test_y_idx = word_to_idx(lang, train_x, train_y, valid_x, valid_y, test_x, test_y)

    n_instances = len(train_x)
    order = range(n_instances)
    padding_idx = lang.get_word_to_idx()['pad']
    V = lang.get_word_size()
    labels_size = len(label2idx)

    encoder = Encoder(args.embedding_size, args.hidden_size, V, args.dropout)
    decoder = Decoder(args.embedding_size ,args.hidden_size, labels_size)

    # optimizer = optim.Adam(encoder.parameters(),
    #                        decoder.parameters(),
    #                        lr = args.lr)

    encoder_optimizer = optim.Adam(encoder.parameters(),
                                   lr = args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(),
                                   lr = args.lr)
    loss_ = nn.NLLLoss()
    for t in range(args.max_epoch):
        print("t{0}".format(t))
        random.shuffle(list(order))

        start_id = 0
        for start_id in range(0, n_instances, args.batch_size):
            end_id = start_id + args.batch_size if start_id + args.batch_size \
                                                   < n_instances else n_instances
            batch_x, batch_y = get_batch(order[start_id:end_id], train_x_idx, train_y_idx, padding_idx)
            batch_size = end_id - start_id

            # batch_x = batch_x.transpos(0,1)
            encoder_outputs, (h, c) = encoder(batch_x)  #  batch_x batch*max_len, (h,c) 2*batch_size*hidden
            h, c = torch.sum(h, 0).view(batch_size, 1, args.hidden_size), torch.sum(c, 0).view( batch_size, 1, args.hidden_size)

            # print("h {0}, c {1}".format(h,c))
            max_len = len(batch_x[0])
            input = Variable(torch.zeros((batch_size, 1, args.embedding_size)))
            loss = 0
            encoder.zero_grad()
            decoder.zero_grad()

            for i in range(max_len):  # cal over loss
                print("i{0}".format(i))
                h_c = (h, c)
                output, (h, c), label = decoder.forward(input, h_c)
                # sum
                label = label.unqueeze(1)  # delete dimension 1

                loss += loss_()
                # output -> batch_y.transpose(0,1)[i]
                # loss += loss_(output, batch_y.transpose(0,1)[i])
                # input = Variable(torch.LongTensor(batch_y.transpose(0,1)[i]))

            # loss.backward()
            # optimizer.step()



if __name__ == '__main__':
    main()