#! /usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'uniphix'
# print ('****************************to be implemented********************************')


def evaluate(gold, predicted):
  '''
  详情可以百度,试着取解释这几个值:
    tp: 分词正确的
    fn: gold中未被预测出的
    fp: 预测错误的
    事实上这几个意义和fn,fp并不能很好吻合
    :param gold:
    :param predicted:
    :return:
  '''
  assert len(gold) == len(predicted)
  tp = 0
  fp = 0
  fn = 0
  for i in range(len(gold)):
    gold_intervals = get_intervals(gold[i])
    predicted_intervals = get_intervals(predicted[i])
    seg = set()
    for interval in gold_intervals:
      seg.add(interval)
      fn += 1
    for interval in predicted_intervals:
      if (interval in seg):
        tp += 1
        fn -= 1
      else:
        fp += 1
  P = 0 if tp == 0 else 1.0 * tp / (tp + fp) #和gold比较的召回率
  R = 0 if tp == 0 else 1.0 * tp / (tp + fn) #分词器标注结果的精度
  F = 0 if P * R == 0 else 2.0 * P * R / (P + R)
  return P, R, F


def get_intervals(tag):
  #tag = tag.data
  intervals = []
  l = len(tag)
  i = 0
  while (i < l):
    if (tag[i].data[0] == 2 or tag[i].data[0] == 3):
      intervals.append((i, i))
      i += 1
      continue
    j = i + 1
    while (True):
      if (j == l or tag[j].data[0] == 0 or tag[j].data[0] == 3):
        intervals.append((i, j - 1))
        i = j
        break
      elif (tag[j].data[0] == 2):
        intervals.append((i, j))
        i = j + 1
        break
      else:
        j += 1
  return intervals