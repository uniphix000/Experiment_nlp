# coding :utf-8
import codecs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

label2idx = {'B':0, 'I':1, 'E':2, 'S':3}
idx2label = {idx:word for word, idx in label2idx.items()}
oov = 1

class Lang:
    def __init__(self):
        super(Lang, self).__init__()
        self.word2idx = {'pad':0, 'oov':1}
        self.idx2word = {}
        self.wordsize = 2

    def add_word(self, word):
        '''
        add the word to dict
        :param word:
        :return:
        '''
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.wordsize += 1

    def add_sentence(self, sentence):
        '''
        add the sentence to dict
        :param sentence:
        :return:
        '''

        for word in sentence:
            self.add_word(word)

    def get_word_size(self):
        '''
        return the dict size
        :return:
        '''

        return self.wordsize

    def get_idx_to_word(self):
        '''
        return self.idx2word
        :return:
        '''
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}
        return self.idx2word

    def get_word_to_idx(self):
        '''
        return self.word2idx
        :return:
        '''
        return self.word2idx


def read_corpus(path):
    '''
    input:path
    output:[我 去 北 京]，[B I E S]
    :param path:
    :return:
    '''
    data_x, data_y = [], []

    with codecs.open(path, 'r', encoding='utf-8') as fin:
        sentences_lines = fin.read().strip().split('\n')
        for index_sentece, value_sentence in enumerate(sentences_lines):
            sentence_x, sentence_label = value_sentence.split('\t')
            data_x.append(sentence_x.strip().split())
            data_y.append(sentence_label.strip().split())

    return data_x, data_y


def data_loader(train_path, valid_path, test_path):
    '''
    input:[[我 去 北 京 B I E S]]
    output:[我 去 北 京]，[B I E S]
    :param train_path:
    :param valid_path:
    :param test_path:
    :return:
    '''

    train_x, train_y = read_corpus(train_path)
    valid_x, valid_y = read_corpus(valid_path)
    test_x, test_y = read_corpus(test_path)

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def generate_dict(lang, train_x):
    '''
    return dict
    :param lang:
    :param train_x:
    :return:
    '''
    for sentences in train_x:
        for word in sentences:
            lang.add_word(word)

    return lang

def word_2_idx(lang, data_x, data_y):
    '''
    conver data_x, data_y to id representation
    [我 爱 北京]
    [0 1 2]
    :param data_x:
    :param data_y:
    :return:
    '''
    data_x_idx = [[lang.get_word_to_idx().get(word, oov) for word in sentence] for sentence in data_x]
    data_y_idx = [[label2idx[word] for word in sentence] for sentence in data_y]

    return data_x_idx, data_y_idx

def word_to_idx(lang, train_x, train_y, valid_x, valid_y, test_x, test_y):
    '''
    convert word and label to idx representation
    input: [我 爱 北京]
    output:[0 1 2]
    :param lang
    :param train_x:
    :param train_y:
    :param valid_x:
    :param valid_y:
    :param test_x:
    :param test_y:
    :return:
    '''
    train_x_idx, train_y_idx = word_2_idx(lang ,train_x, train_y)
    valid_x_idx, valid_y_idx = word_2_idx(lang, valid_x, valid_y)
    test_x_idx, test_y_idx = word_2_idx(lang, test_x, test_y)

    return train_x_idx, train_y_idx,  valid_x_idx, valid_y_idx, test_x_idx, test_y_idx

def padding(data, padding_idx = 0):
    '''
    padding for data
    :param data:
    :param padding_idx:
    :return:
    '''
    batch_data = []
    max_len = max([len(sentence) for sentence in data])
    batch_data = [sentence + (max_len-len(sentence))*[padding_idx] for sentence in data]

    return batch_data

def get_batch(order_list, data_x_idx, data_y_idx, padding_idx = 0):
    '''
    get batch_data
    :param order_list:
    :param data_x_idx:
    :param data_y_idx:
    :param padding_idx:
    :return:
    '''
    lens = []
    batch_x = [data_x_idx[i] for i in order_list]
    batch_y = [data_y_idx[i] for i in order_list]

    batch_x = padding(batch_x, padding_idx)
    batch_y = padding(batch_y, padding_idx)

    batch_x = Variable(torch.LongTensor(batch_x))
    batch_y = Variable(torch.LongTensor(batch_y))


    return batch_x, batch_y
