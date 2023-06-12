#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/6/9 13:37
# @Author  : Linxuan Jiang
# @File    : utils_re.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT

import torch
import torch.nn as nn
import math
import re

from transformers import BertTokenizer, BertModel
import json
import numpy as np


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
    tokenizer = BertTokenizer.from_pretrained("checkpoint/Bert_embeading")

    id2predicate = {}
    predicate2id = {}


class Model4s(nn.Module):
    def __init__(self, hidden_size=768):
        super(Model4s, self).__init__()
        self.bert = BertModel.from_pretrained(config.PATH_BERT)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=hidden_size, out_features=2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, input_mask, segment_ids, hidden_size=768):
        hidden_states = self.bert(input_ids,
                                  attention_mask=input_mask,
                                  token_type_ids=segment_ids)[0]  # (batch_size, sequence_length, hidden_size)
        output = self.sigmoid(self.linear(self.dropout(hidden_states))).pow(2)
        return output, hidden_states


class Model4po(nn.Module):
    def __init__(self, num_p=config.num_p, hidden_size=768):
        super(Model4po, self).__init__()
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_p * 2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, batch_subject_ids, input_mask):
        """

        :param hidden_states: [1,256,768]
        :param batch_subject_ids: [6,9]
        :param input_mask: [1 if in length else 0]
        :return:
        """
        all_s = torch.zeros((hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]),
                            dtype=torch.float32)

        """
        根据subject 对记忆进行增强，简单做法是【Batch ， 字数： 768】 = 768 （广播）
        hidden_states = hidden_states + Bias
        其中Bias 由subject 位置信息和句子长度input_mask所影响，
        """
        for b in range(hidden_states.shape[0]):
            s_start = batch_subject_ids[b][0]
            s_end = batch_subject_ids[b][1]
            s = hidden_states[b][s_start] + hidden_states[b][s_end]  # s定义subject的位置信息【768】
            cue_len = torch.sum(input_mask[b])
            all_s[b, :cue_len, :] = s
        hidden_states += all_s

        output = self.sigmoid(self.linear(self.dropout(hidden_states))).pow(4) # 1,256,46

        return output  # (batch_size, max_seq_len, num_p*2)


def load_model():
    load_schema(config.PATH_SCHEMA)
    checkpoint = torch.load(config.PATH_MODEL, map_location='cpu')

    # model4s = Model4s()
    # model4s.load_state_dict(checkpoint['model4s_state_dict'])
    # # model4s.cuda()
    #
    # model4po = Model4po()
    # model4po.load_state_dict(checkpoint['model4po_state_dict'])
    # model4po.cuda()

    model4s = Model4s()
    model4s.load_state_dict(checkpoint['model4s_state_dict'], False)

    model4po = Model4po()
    model4po.load_state_dict(checkpoint['model4po_state_dict'], False)

    return model4s, model4po


def extract_spoes(text, model4s, model4po):
    """
    return: a list of many tuple of (s, p, o)
    """
    # 处理text
    with torch.no_grad():
        tokenizer = config.tokenizer
        max_seq_len = config.max_seq_len
        token_ids = torch.tensor(
            tokenizer.encode(text, max_length=max_seq_len, pad_to_max_length=True, add_special_tokens=True)).view(1, -1)
        if len(text) > max_seq_len - 2:
            text = text[:max_seq_len - 2]
        mask_ids = torch.tensor([1] * (len(text) + 2) + [0] * (max_seq_len - len(text) - 2)).view(1, -1)
        segment_ids = torch.tensor([0] * max_seq_len).view(1, -1)
        subject_labels_pred, hidden_states = model4s(token_ids, mask_ids, segment_ids)
        subject_labels_pred = subject_labels_pred.cpu()
        subject_labels_pred[0, len(text) + 2:, :] = 0
        start = np.where(subject_labels_pred[0, :, 0] > 0.4)[0]
        end = np.where(subject_labels_pred[0, :, 1] > 0.4)[0]

        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))

        if len(subjects) == 0:
            return []
        subject_ids = torch.tensor(subjects).view(1, -1)

        spoes = []
        for s in subjects:
            object_labels_pred = model4po(hidden_states, subject_ids, mask_ids)
            object_labels_pred = object_labels_pred.view((1, max_seq_len, config.num_p, 2)).cpu()
            object_labels_pred[0, len(text) + 2:, :, :] = 0
            start = np.where(object_labels_pred[0, :, :, 0] > 0.4)
            end = np.where(object_labels_pred[0, :, :, 1] > 0.4)

            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append((s, predicate1, (_start, _end)))
                        break

    id_str = ['[CLS]']
    i = 1
    index = 0
    while i < token_ids.shape[1]:
        if token_ids[0][i] == 102:
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


def get_triples(content, model4s, model4po):
    if len(content) == 0:
        return []
    text_list = content.split('。')[:-1]
    res = []
    for text in text_list:
        if len(text) > 128:
            text = text[:128]
        triples = extract_spoes(text, model4s, model4po)
        res.append({
            'text': text,
            'triples': triples
        })
    return res


def load_schema(path):
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


if __name__ == "__main__":
    path = "checkpoint/medical_re/bert_embeading/predicate.json"

    load_schema(path)
    model4s, model4po = load_model()

    text = '据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人。'  # content是输入的一段文字
    res = get_triples(text, model4s, model4po)
    print(json.dumps(res, ensure_ascii=False, indent=True))
