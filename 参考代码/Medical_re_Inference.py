#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/6/9 11:47
# @Author  : Linxuan Jiang
# @File    : Medical_re_Inference.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright (MIT) 2023 - 2023 Linxuan Jiang, Inc. All Rights Reserved


import codecs

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import json
from tools.utils import load_vocab
import pandas as pd
from All_models.model_ner.bert_lstm_crf import BERT_LSTM_CRF
import os
import math
import re
import torch.nn as nn
import numpy as np
from tools.utils_re import Model4s, Model4po
from transformers import BertTokenizer


class config:
    batch_size = 32
    max_seq_len = 256
    num_p = 23
    learning_rate = 1e-5
    EPOCH = 2

    # PATH_SCHEMA = "/Users/yangyf/workplace/model/medical_re/predicate.json"
    # PATH_TRAIN = '/Users/yangyf/workplace/model/medical_re/train_data.json'
    # PATH_BERT = "/Users/yangyf/workplace/model/medical_re/"
    # PATH_MODEL = "/Users/yangyf/workplace/model/medical_re/model_re.pkl"
    # PATH_SAVE = '/content/model_re.pkl'
    # tokenizer = BertTokenizer.from_pretrained("/Users/yangyf/workplace/model/medical_re/" + 'vocab.txt')

    PATH_SCHEMA = "checkpoint/medical_re/bert_embeading/predicate.json"
    PATH_TRAIN = 'model_re/train/data/train_data.json'
    PATH_BERT = "checkpoint/Bert_embeading"
    PATH_MODEL = "checkpoint/medical_re/model_re.pkl"
    PATH_SAVE = 'checkpoint/medical_re'
    tokenizer = BertTokenizer.from_pretrained("checkpoint/medical_re/bert_embeading/vocab.txt")

    id2predicate = {0: '相关疾病', 1: '相关症状', 2: '临床表现', 3: '检查', 4: '用法用量', 5: '贮藏',
                    6: '性状', 7: '禁忌', 8: '不良反应', 9: '有效期', 10: '注意事项', 11: '适应症',
                    12: '成份', 13: '病因', 14: '治疗', 15: '规格', 16: '并发症', 17: '药物相互作用',
                    18: '主治', 19: '入药部位', 20: '功效', 21: '性味', 22: '功能主治'}

    predicate2id = {'相关疾病': 0, '相关症状': 1, '临床表现': 2, '检查': 3, '用法用量': 4, '贮藏': 5,
                    '性状': 6, '禁忌': 7, '不良反应': 8, '有效期': 9, '注意事项': 10, '适应症': 11,
                    '成份': 12, '病因': 13, '治疗': 14, '规格': 15, '并发症': 16, '药物相互作用': 17,
                    '主治': 18, '入药部位': 19, '功效': 20, '性味': 21, '功能主治': 22}


class RE_Inference:
    def __init__(self, config):
        self.init_schema(config.PATH_SCHEMA)
        self.init_model(config.PATH_MODEL)

    def init_schema(self, path):

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
            predicate = list(data.keys())
            prediction2id = {}
            id2predicate = {}
            for i in range(len(predicate)):
                prediction2id[predicate[i]] = i
                id2predicate[i] = predicate[i]
        num_p = len(predicate)
        config.prediction2id = prediction2id
        config.id2predicate = id2predicate
        config.num_p = num_p

    def init_model(self, checkpoint):
        self.model4s = Model4s()
        self.model4po = Model4po()
        if checkpoint:
            wight = torch.load(checkpoint, map_location='cpu')
            self.model4s.load_state_dict(wight['model4s_state_dict'], False)
            self.model4po.load_state_dict(wight['model4po_state_dict'], False)

    def get_triples(self, content):
        if len(content) == 0:
            return []
        text_list = content.split('。')[:-1]
        res = []
        for text in text_list:
            if len(text) > 128:
                text = text[:128]
            triples = self.extract_spoes(text)
            res.append({
                'text': text,
                'triples': triples
            })
        return res

    def extract_spoes(self, text):
        """
        return: a list of many tuple of (s, p, o)
        """
        # 处理text
        with torch.no_grad():
            tokenizer = config.tokenizer
            max_seq_len = config.max_seq_len  # 256
            token_ids = torch.tensor(
                tokenizer.encode(text, max_length=max_seq_len, pad_to_max_length=True, add_special_tokens=True)).view(1,
                                                                                                                      -1)
            if len(text) > max_seq_len - 2:
                text = text[:max_seq_len - 2]

            # mask_id 给padding 添加mask 预测的结果丢弃
            mask_ids = torch.tensor([1] * (len(text) + 2) + [0] * (max_seq_len - len(text) - 2)).view(1, -1)
            segment_ids = torch.tensor([0] * max_seq_len).view(1, -1)
            subject_labels_pred, hidden_states = self.model4s(token_ids, mask_ids, segment_ids)
            # hidden_states：[1,256,768]  subject_labels_pred: [1,256,2]

            subject_labels_pred = subject_labels_pred.cpu()
            subject_labels_pred[0, len(text) + 2:, :] = 0
            start = np.where(subject_labels_pred[0, :, 0] > 0.4)[0]  # [ 1, 256] 预测start ：6， 阈值设定为0.4
            end = np.where(subject_labels_pred[0, :, 1] > 0.4)[0]  # [1,256] 预测end：9

            subjects = []  # subjects 个数由start个数决定， 结束标志定义为与比start的值大且最近的那个end
            for i in start:
                j = end[end >= i]
                if len(j) > 0:
                    j = j[0]
                    subjects.append((i, j))

            if len(subjects) == 0:
                return []
            subject_ids = torch.tensor(subjects).view(1, -1)

            spoes = []   # [(6,9), 2,(8,9)]  # 实体， 关系类别， 实体
            for s in subjects:
                object_labels_pred = self.model4po(hidden_states, subject_ids, mask_ids)  # [1,256,46]
                object_labels_pred = object_labels_pred.view((1, max_seq_len, config.num_p, 2)).cpu()  # [1,256,23,2]
                object_labels_pred[0, len(text) + 2:, :, :] = 0  # [199, 23, 2] , padding 之后的值置零
                # print(object_labels_pred[0, :, :, 0].shape) # [256, 23]
                start = np.where(object_labels_pred[0, :, :, 0] > 0.4)  # 第一个维度上大于0.4的值的行索引和列索引
                # [8,15,18,27,30,41,51], [2,2,2,2,2,13,13]
                end = np.where(object_labels_pred[0, :, :, 1] > 0.4)  # 第一个维度上大于0.4的值的行索引和列索引
                for _start, predicate1 in zip(*start):
                    for _end, predicate2 in zip(*end):
                        if _start <= _end and predicate1 == predicate2:  # 同一个类别
                            spoes.append((s, predicate1, (_start, _end)))
                            break

        id_str = ['[CLS]']
        i = 1
        index = 0
        while i < token_ids.shape[1]:
            if token_ids[0][i] == 102:  # 102 Cls？？
                break

            word = tokenizer.decode(token_ids[0, i:i + 1])
            word = re.sub('#+', '', word)
            if word != '[UNK]':
                id_str.append(word)
                index += len(word)
                i += 1
            else:
                j = i + 1
                while j < token_ids.shape[1]:
                    if token_ids[0][j] == 102:
                        break
                    word_j = tokenizer.decode(token_ids[0, j:j + 1])
                    if word_j != '[UNK]':
                        break
                    j += 1
                if token_ids[0][j] == 102 or j == token_ids.shape[1]:
                    while i < j - 1:
                        id_str.append('')
                        i += 1
                    id_str.append(text[index:])
                    i += 1
                    break
                else:
                    index_end = text[index:].find(word_j)
                    word = text[index:index + index_end]
                    id_str.append(word)
                    index += index_end
                    i += 1

        res = []
        for s, p, o in spoes:
            s_start = s[0]
            s_end = s[1]
            sub = ''.join(id_str[s_start:s_end + 1])
            o_start = o[0]
            o_end = o[1]
            obj = ''.join(id_str[o_start:o_end + 1])
            res.append((sub, config.id2predicate[p], obj))

        return res


if __name__ == "__main__":
    text = r'据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人。'
    Intity = RE_Inference(config)

    print(Intity.get_triples(text))

    """ [{'text': '据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人',
      'triples': [('新冠肺炎', '临床表现', '肺炎'), ('新冠肺炎', '临床表现', '发热'), ('新冠肺炎', '临床表现', '咳嗽'),
                  ('新冠肺炎', '临床表现', '胸闷'), ('新冠肺炎', '临床表现', '乏力'),
                  ('新冠肺炎', '病因', '自身免疫系统缺陷'), ('新冠肺炎', '病因', '人传人')]}]"""
