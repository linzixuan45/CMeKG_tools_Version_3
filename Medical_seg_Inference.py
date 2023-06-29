#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/6/12 15:09
# @Author  : Linxuan Jiang
# @File    : Medical_cws_Inference.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT
# coding:utf-8
import codecs

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from All_models.model_cws.bert_lstm_crf import BERT_LSTM_CRF

model_checkpoint = 'checkpoint/Pretrain_weight/medical_cws/pytorch_model.pkl'
vocab_url = 'checkpoint/Pretrain_weight/Bert_embedding/vocab.txt'
Bert_Embedding = 'checkpoint/Pretrain_weight/Bert_embedding'

l2i_dic = {"S": 0, "B": 1, "M": 2, "E": 3, "<pad>": 4, "<start>": 5, "<eos>": 6}

i2l_dic = {0: "S", 1: "B", 2: "M", 3: "E", 4: "<pad>", 5: "<start>", 6: "<eos>"}

# 超参
max_length = 150
batch_size = 200
epochs = 500
tagset_size = len(l2i_dic)
use_cuda = False


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


class MedicalSeg:
    def __init__(self):
        self.NEWPATH = model_checkpoint
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
            self.use_cuda = True
        else:
            self.device = torch.device("cpu")
            self.use_cuda = False

        self.vocab = load_vocab(vocab_url)
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        self.model = BERT_LSTM_CRF(Bert_Embedding, tagset_size, 768, 200, 2,
                                   dropout_ratio=0.5, dropout1=0.5, use_cuda=use_cuda)

        if use_cuda:
            self.model.cuda()

    def from_input(self, input_str):
        # 单行的输入
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        text = ['[CLS]'] + [x for x in input_str] + ['[SEP]']
        raw_text.append(text)
        cur_len = len(text)
        # raw_textid = [self.vocab[x] for x in text] + [0] * (max_length - cur_len)
        raw_textid = [self.vocab[x] for x in text if self.vocab.__contains__(x)] + [0] * (max_length - cur_len)
        textid.append(raw_textid)
        raw_textmask = [1] * cur_len + [0] * (max_length - cur_len)
        textmask.append(raw_textmask)
        textlength.append([cur_len])
        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength

    def from_txt(self, input_path):
        # 多行输入
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line) > 148:
                    line = line[:148]
                temptext = ['[CLS]'] + [x for x in line[:-1]] + ['[SEP]']
                cur_len = len(temptext)
                raw_text.append(temptext)

                tempid = [self.vocab[x] for x in temptext[:cur_len]] + [0] * (max_length - cur_len)
                textid.append(tempid)
                textmask.append([1] * cur_len + [0] * (max_length - cur_len))
                textlength.append([cur_len])

        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength

    def recover_to_text(self, pred, raw_text):
        # 输入[标签list]和[原文list],batch为1
        pred = [i2l_dic[t.item()] for t in pred[0]]
        pred = pred[:len(raw_text)]
        pred = pred[1:-1]
        raw_text = raw_text[1:-1]
        raw = ""
        res = ""
        for tag, char in zip(pred, raw_text):
            res += char
            if tag in ["S", 'E']:
                res += ' '
            raw += char
        return raw, res

    def predict_sentence(self, sentence):
        if sentence == '':
            print("输入为空！请重新输入")
            return
        if len(sentence) > 148:
            print("输入句子过长，请输入小于148的长度字符！")
            sentence = sentence[:148]
        raw_text, test_ids, test_masks, test_lengths = self.from_input(sentence)

        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location={'cuda:0': 'cpu'}), False)
        # self.model.load_state_dict(torch.load(self.NEWPATH, map_location={'cuda:0': 'cpu'}), False)
        self.model.eval()

        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if use_cuda:
                sentence = sentence.cuda()
                masks = masks.cuda()

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()

            raw, res = self.recover_to_text(predict_tags, batch_raw_text)
            # print("输入：", raw)
            # print("结果：", res)
        return res

    def predict_file(self, input_file, output_file):
        # raw_text, test_ids, test_masks, test_lengths = self.from_txt("./data/raw_text.txt")
        raw_text, test_ids, test_masks, test_lengths = self.from_txt(input_file)

        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location={'cuda:0': 'cpu'}))
        self.model.eval()

        op_file = codecs.open(output_file, 'w', 'utf-8')
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if use_cuda:
                sentence = sentence.cuda()
                masks = masks.cuda()

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()

            raw, res = self.recover_to_text(predict_tags, batch_raw_text)
            op_file.write(res + '\n')

        op_file.close()
        print('处理完成！')
        print("results have been stored in {}".format(output_file))


if __name__ == "__main__":
    medical_seg = MedicalSeg()

    res = medical_seg.predict_sentence(
        "肾上腺由皮质和髓质两个功能不同的内分泌器官组成，皮质分泌肾上腺皮质激素，最好的皮质就是牛皮,骨折的患者都是去骨科医院治疗")
    print(res)
    """
    肾上腺 由 皮质 和 髓质 两 个 功能 不 同 的 内分泌器官 组成 ，
    皮质 分泌 肾上腺皮质激素 ，最好的皮质 就是 牛皮,骨折 的 患者 都是 去 骨科 医院 治疗 
    """