#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/6/12 12:28
# @Author  : Linxuan Jiang
# @File    : Trainer_re.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT


import random
import json
import numpy as np
import torch
import torch.nn as nn
# from constant import ProductionConfig as Path
from transformers import BertTokenizer, BertModel
from itertools import cycle
import gc
import random
import time
import re
from All_models.model_re.medical_re import Model4s, Model4po
from All_models.Optimizer import Lion
from dataset.dataload_re.dataset import ReDataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


class config:
    batch_size = 32
    max_seq_len = 256
    num_p = 23
    learning_rate = 1e-5
    EPOCH = 2
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


class Trainer:
    def __init__(self):
        self.checkpoint = config.PATH_MODEL
        self.medical_bert = 'model_ner/medical_ner'
        self.save_model_dir = config.PATH_SAVE
        self.max_length = 450
        self.batch_size = 32
        self.epochs = config.EPOCH
        self.use_cuda = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """--------------------------"""
        self.init_dataset()
        self.init_model()
        self.train()

    def log(self, msg):
        msg = " {}:     {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), msg)
        print(msg)

    def init_model(self):
        self.model4s = Model4s(config)
        self.model4po = Model4po(config)
        if self.checkpoint:
            self.log(f'Restoring from checkpoint: {self.checkpoint}')
            wight = torch.load(self.checkpoint, map_location='cpu')
            self.model4s.load_state_dict(wight['model4s_state_dict'], False)
            self.model4po.load_state_dict(wight['model4po_state_dict'], False)

        if self.use_cuda:
            self.model4s = self.model4s.to(self.device)
            self.model4po = self.model4po.to(self.device)

        param_optimizer = list(self.model4s.named_parameters()) + list(self.model4po.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = Lion(optimizer_grouped_parameters)  # 比AdamW更快\
        self.loss = nn.BCELoss(reduction='none')

    def loss_func(self, pred, target, mask_ids):
        temp_loss = self.loss(pred, target)
        temp_loss = torch.mean(temp_loss, dim=2, keepdim=False) * mask_ids
        temp_loss = torch.sum(temp_loss)
        temp_loss = temp_loss / torch.sum(mask_ids)
        return temp_loss

    def init_dataset(self):
        train_data = ReDataset()
        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size, pin_memory=True)

    def train(self):
        best_f = -100
        for epoch in range(self.epochs):
            print('epoch: {}，train'.format(epoch))
            for i, train_batch in enumerate(tqdm(self.train_loader)):
                token_ids, mask_ids, segment_ids, subject_labels, subject_ids, object_labels = train_batch

                subject_labels_pred, hidden_states = self.model4s(token_ids, mask_ids, segment_ids)
                object_labels_pred = self.model4po(hidden_states, subject_ids, mask_ids)

                loss4s = self.loss_func(subject_labels_pred,subject_labels, mask_ids)
                loss4po = self.loss_func(object_labels_pred, object_labels, mask_ids)
                loss = loss4s + loss4po

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


