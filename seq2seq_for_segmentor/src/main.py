# -*- coding: utf-8 -*-
import argparse
from dataloader import *
import random
from module import *
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
from utils import *
import itertools

use_cuda = torch.cuda.is_available()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')

def main():
    cmd = argparse.ArgumentParser("seq2seq for segmentor")
    cmd.add_argument("--train_input_path", help = "path of input ", default='../data/train.txt')
    cmd.add_argument("--test_input_path", help="path of test ", default='../data/test.txt')
    cmd.add_argument("--valid_input_path", help="path of valid ", default='../data/valid.txt')
    cmd.add_argument("--max_epoch", help = "epoch for training", default=20)
    cmd.add_argument("--embedding_size", help="embedding for training", default=100)
    cmd.add_argument("--hidden_size", help="hidden for training", default=128)
    cmd.add_argument("--lr", help = "learning rato for training", default=0.001)
    cmd.add_argument("--batch_size", help = "batch_size for training", default=4)
    cmd.add_argument("--dropout", help="dropout for training", default=0.75)
    cmd.add_argument("--seed", help = "seed for training", default=1)
    cmd.add_argument("--optimizer",help = "optimizer for training", default='adam')

    args = cmd.parse_args()

    train_x, train_y, valid_x, valid_y, test_x, test_y = data_loader(args.train_input_path, args.test_input_path, args.valid_input_path)
    logging.info('training_data_size = {0}, valid_data_size = {1}, test_data_size = {2}'.format(len(train_x), len(valid_x), len(test_x)))
    train_x, train_y = test_x, test_y
    lang = Lang()
    lang = generate_dict(lang, train_x)
    train_x_idx, train_y_idx,  valid_x_idx, valid_y_idx, test_x_idx, test_y_idx = word_to_idx(lang, train_x, train_y, valid_x, valid_y, test_x, test_y)

    logging.info("dict generated! dict size {0}".format(lang.get_word_size()))


    n_instances = len(train_x)
    order = range(n_instances)
    padding_idx = lang.get_word_to_idx()['pad']
    V = lang.get_word_size()
    labels_size = len(label2idx)

    encoder = Encoder(args.embedding_size, args.hidden_size, V, args.dropout)
    decoder = Decoder(args.embedding_size ,args.hidden_size, labels_size)
    encoder = encoder.cuda() if use_cuda else encoder
    decoder = decoder.cuda() if use_cuda else decoder

    # optimizer = optim.Adam(encoder.parameters(),
    #                        decoder.parameters(),
    #                        lr = args.lr)

    encoder_optimizer = optim.Adam(encoder.parameters(),
                                   lr = args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(),
                                   lr = args.lr)
    loss_ = nn.NLLLoss()


    best_valid_F, best_test_F = 0, 0
    for t in range(args.max_epoch):
        random.shuffle(list(order))
        #print ('==================',order[0:5])
        start_id = 0
        for start_id in range(0, n_instances, args.batch_size):
            end_id = start_id + args.batch_size if start_id + args.batch_size \
                                                   < n_instances else n_instances
            batch_x, batch_y = get_batch(order[start_id:end_id], train_x_idx, train_y_idx, padding_idx)
            batch_size = end_id - start_id

            encoder_outputs, (h, c) = encoder(batch_x)  #  batch_x batch*max_len, (h,c) 2*batch_size*hidden
            h, c = torch.sum(h, 0).view(1, batch_size, args.hidden_size), torch.sum(c, 0).view(1, batch_size, args.hidden_size)

            max_len = len(batch_x[0])
            input = Variable(torch.zeros((batch_size, 1, args.embedding_size))).cuda() if use_cuda else \
                Variable(torch.zeros((batch_size, 1, args.embedding_size)))
            loss = 0
            encoder.zero_grad()
            decoder.zero_grad()

            for i in range(max_len):  # cal over loss
                h_c = (h, c)
                output, (h, c), label = decoder.forward(input, h_c)  # output: (batch_size, 1, hidden_size), label:
                                                                     # (batch_size, 1, label_size) h:(1,batch_size, hidden_size)
                # sum
                label = F.log_softmax(label.view(batch_size, -1))  # 每列与tag作loss, (batch_size, label_size)
                gold = batch_y.transpose(0,1)[i]
                non_pad_ele_idx = []
                idx_count = 0
                for ele in gold:
                    if ele.data[0] != 4:
                        non_pad_ele_idx.append(idx_count)
                    idx_count += 1
                non_pad_ele_idx = Variable(torch.LongTensor(non_pad_ele_idx)).cuda() if use_cuda else \
                    Variable(torch.LongTensor(non_pad_ele_idx))
                #print (torch.index_select(label, 0, non_pad_ele_idx))
                loss += loss_(torch.index_select(label, 0, non_pad_ele_idx), torch.index_select(gold, 0, non_pad_ele_idx))  # batch_y: (batch_size, max_length)
                # output -> batch_y.transpose(0,1)[i]
                # loss += loss_(output, batch_y.transpose(0,1)[i])
                # input = Variable(torch.LongTensor(batch_y.transpose(0,1)[i]))
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

        logging.info('--------------------Round {0}---------------------'.format(t))
        valid_F = eval_model(encoder, decoder, valid_x_idx, valid_y_idx, args.batch_size, args.embedding_size, padding_idx)
        if (valid_F > best_valid_F):
            test_F = eval_model(encoder, decoder, test_x_idx, test_y_idx, args.batch_size, args.embedding_size, padding_idx)
            logging.info("New Record!!! In Round {0}: test F: {1}%".format(t, test_F*100))
            best_test_F = max(best_test_F, test_F)
        best_valid_F = max(best_valid_F, valid_F)
        logging.info('Round {0} ended: valid F: {1}'.format(t, valid_F*100))
    logging.info('Training Completed! best valid F: {0}; best test F: {1}'.format(best_valid_F*100, best_test_F*100))




def eval_model(encoder, decoder, valid_x_idx, valid_y_idx, batch_size, embed_size, pad_idx):
    '''

    :param model:
    :param valid_x_idx:
    :param valid_y_idx:
    :param batch_size:
    :param pad_idx:
    :return:
    '''
    encoder.eval()
    decoder.eval()
    x_size = len(valid_x_idx)
    order = range(x_size)
    random.shuffle(list(order))

    start_id = 0
    predict_all = []
    gold_all = []
    for start_id in range(0, x_size, batch_size):
        end_id = start_id + batch_size if start_id + batch_size \
                                               < x_size else x_size
        batch_x, batch_y = get_batch(order[start_id:end_id], valid_x_idx, valid_y_idx, pad_idx)
        batch_size = end_id - start_id

        # batch_x = batch_x.transpos(0,1)
        encoder_outputs, (h, c) = encoder(batch_x)  #  batch_x batch*max_len, (h,c) 2*batch_size*hidden
        h, c = torch.sum(h, 0).view(1, batch_size, -1), torch.sum(c, 0).view(1, batch_size,  -1)

        # print("h {0}, c {1}".format(h,c))
        max_len = len(batch_x[0])
        input = Variable(torch.zeros((batch_size, 1, embed_size))).cuda() if use_cuda else \
            Variable(torch.zeros((batch_size, 1, embed_size)))

        predict_result_box = []
        non_pad_ele_idx_box = []
        for i in range(max_len):  # cal over loss
            h_c = (h, c)
            output, (h, c), label = decoder.forward(input, h_c)  # output: (batch_size, 1, hidden_size), label:
                                                                 # (batch_size, 1, label_size) h:(1,batch_size, hidden_size)
            # sum
            label = F.log_softmax(label.view(batch_size, -1))  # (batch_size, label_size)
            _ , max_id= torch.max(label, 1)
            gold = batch_y.transpose(0,1)[i]
            non_pad_ele_idx = []
            idx_count = 0
            for ele in gold:
                if ele.data[0] != 4:
                    non_pad_ele_idx.append(idx_count)  # 挑选出非pad的序号
                idx_count += 1
            non_pad_ele_idx = Variable(torch.LongTensor(non_pad_ele_idx)).cuda() if use_cuda else Variable(torch.LongTensor(non_pad_ele_idx))
            predict_result_box.append(max_id)
            non_pad_ele_idx_box.append(non_pad_ele_idx)  # 格式[[],]
        predict, gold = prepare_for_evaluate(predict_result_box, batch_y, non_pad_ele_idx_box)
        predict_all.append(predict)
        gold_all.append(gold)
    _, _, F_Score = evaluate(gold_all, predict_all)

    return F_Score






def flatten(lst):
    return list(itertools.chain.from_iterable(lst))

def prepare_for_evaluate(predict_result_box, batch_y, non_pad_ele_idx_box):
    """
    得到预测的长串,即[token_size]
    :param predict_result_box: 列形式的预测结果 [] max_length个batch_size
    :param batch_y:  被padding的y  [[],] batch_size个max_length
    :param non_pad_ele_idx_box: [[],[]] max_length个batch_size,记录了哪几个元素是可用的
    :return:  predict:[[],] gold:[[],]
    """
    max_length = len(predict_result_box)
    batch_size = len(batch_y)
    predict = [[] for i in range(batch_size)]
    gold = [[] for i in range(batch_size)]
    for i in range(max_length):
        for j in range(batch_size):
            if (j in non_pad_ele_idx_box[i].data):
                predict[j].append(predict_result_box[i][j])
                gold[j].append(batch_y[j][i])

    return flatten(predict), flatten(gold)








if __name__ == '__main__':
    main()