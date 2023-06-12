# coding:utf-8
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

# tag-entity:{d:疾病 s:临床表现 b:身体 e:医疗设备 p:医疗程序 m:微生物类 k:科室 i:医学检验项目 y:药物}
l2i_dic = {"o": 0, "d-B": 1, "d-M": 2, "d-E": 3, "s-B": 4, "s-M": 5, "s-E": 6,
           "b-B": 7, "b-M": 8, "b-E": 9, "e-B": 10, "e-M": 11, "e-E": 12, "p-B": 13, "p-M": 14, "p-E": 15,
           "m-B": 16, "m-M": 17,
           "m-E": 18, "k-B": 19, "k-M": 20, "k-E": 21, "i-B": 22, "i-M": 23, "i-E": 24, "y-B": 25, "y-M": 26,
           "y-E": 27, "<pad>": 28, "<start>": 29, "<eos>": 30}

i2l_dic = {0: "o", 1: "d-B", 2: "d-M", 3: "d-E", 4: "s-B", 5: "s-M",
           6: "s-E", 7: "b-B", 8: "b-M", 9: "b-E", 10: "e-B", 11: "e-M", 12: "e-E", 13: "p-B", 14: "p-M",
           15: "p-E",
           16: "m-B", 17: "m-M", 18: "m-E", 19: "k-B", 20: "k-M", 21: "k-E",
           22: "i-B", 23: "i-M", 24: "i-E", 25: "y-B", 26: "y-M", 27: "y-E", 28: "<pad>", 29: "<start>",
           30: "<eos>"}

max_length = 450
batch_size = 2
epochs = 100
tagset_size = len(l2i_dic)
use_cuda = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Medical_NER_Infer(object):
    def __init__(self, checkpoint=None):
        self.bert_dir = 'checkpoint/Bert_embeading'
        self.vocab_url = os.path.join(self.bert_dir, 'vocab.txt')
        self.checkpoint = checkpoint
        self.use_cuda = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tagset_size = len(l2i_dic)
        self._init_model()

    def _init_model(self):
        self.vocab = load_vocab(self.vocab_url)
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        self.model = BERT_LSTM_CRF(self.bert_dir, self.tagset_size, 768, 200, 2,
                                   dropout_ratio=0.5, dropout1=0.5, use_cuda=self.use_cuda)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location={'cuda:0': 'cpu'}), False)
        self.model.eval()
        if self.use_cuda:
            self.model.to(self.device)

    def find_subtext_index(self, text, keywords):
        """

        Args:
            text: 出生后感染性肺炎可出现发热或体温不升
            keywords: keywords = ['发热', '体温不升', '反应差']

        Returns: [[11, 13]]

        """
        index_ls = []
        for keyword in keywords:
            escaped_keyword = re.escape(keyword)
            matches = re.finditer(escaped_keyword, text)
            # matches = re.finditer(keyword, text)
            indices = [match.start() for match in matches]
            for value in indices:
                index_ls.append([value, value + len(keyword)])

        # 使用集合进行去重
        unique_list = [list(x) for x in set(tuple(x) for x in index_ls)]
        unique_list.sort(key=lambda x: x[0])

        return unique_list

    def from_input(self, input_str):
        """
        1。给句子增加开头和结尾标志， 并转为词向量
        """
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
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip()) == 0:
                    continue
                if len(line) > 400:
                    line = line[:400]
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

    def split_entity_input(self, label_seq):
        entity_mark = dict()
        entity_pointer = None
        for index, label in enumerate(label_seq):
            # print(f"before: {label_seq}")
            if label.split('-')[-1] == 'B':
                category = label.split('-')[0]
                entity_pointer = (index, category)
                entity_mark.setdefault(entity_pointer, [label])
            elif label.split('-')[-1] == 'M':
                if entity_pointer is None: continue
                if entity_pointer[1] != label.split('-')[0]: continue
                entity_mark[entity_pointer].append(label)
            elif label.split('-')[-1] == 'E':
                if entity_pointer is None: continue
                if entity_pointer[1] != label.split('-')[0]: continue
                entity_mark[entity_pointer].append(label)
            else:
                entity_pointer = None
        # print(entity_mark)  # index
        return entity_mark

    def predict_sentence(self, sentence):
        """
        加入重叠区域效果是否会更好
        :param sentence:
        :return:
        """
        split_limit = 10
        tag_dic = {"d": "疾病", "b": "身体", "s": "症状", "p": "医疗程序", "e": "医疗设备", "y": "药物", "k": "科室",
                   "m": "微生物类", "i": "医学检验项目"}
        sentence = str(sentence)
        if sentence == '':
            print("输入为空！请重新输入")
            return
        entity_list_ls = []
        if len(sentence) > split_limit:
            juhao_ls = self.find_subtext_index(sentence, ['。', '，', "、", ',', ])
            juhao_ls = [value[1] for value in juhao_ls]
            split_pices = math.ceil(len(sentence) / split_limit)
            split_bias = [0]
            for j in range(split_pices + 1):  # 找到切分的下标也就是偏置
                for i, value in enumerate(juhao_ls):
                    if i + 1 < len(juhao_ls) and juhao_ls[i] < split_limit * j and juhao_ls[i + 1] > split_limit * j:
                        split_bias.append(value)  # value 是需要切分的位置
            split_bias.append(len(sentence))
            split_bias = list(set(split_bias))
            split_bias = sorted(split_bias)
            for i, vlaue in enumerate(split_bias):
                if i + 1 < len(split_bias):
                    temp_text = sentence[split_bias[i]: split_bias[i + 1]]
                    print(temp_text)
                    raw_text, test_ids, test_masks, test_lengths = self.from_input(temp_text)
                    test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
                    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
                    self.model.load_state_dict(torch.load(self.checkpoint, map_location={'cuda:0': 'cpu'}), False)
                    self.model.eval()
                    for i, dev_batch in enumerate(test_loader):
                        temp_text, masks, lengths = dev_batch
                        batch_raw_text = raw_text[i]
                        temp_text, masks, lengths = Variable(temp_text), Variable(masks), Variable(lengths)
                        if use_cuda:
                            temp_text = temp_text.to(device)
                            masks = masks.to(device)

                        predict_tags = self.model(temp_text, masks)
                        predict_tags.tolist()
                        predict_tags = [i2l_dic[t.item()] for t in predict_tags[0]]
                        predict_tags = predict_tags[:len(batch_raw_text)]
                        pred = predict_tags[1:-1]
                        raw_text = batch_raw_text[1:-1]
                        entity_mark = self.split_entity_input(pred)
                        entity_list = {}
                        if entity_mark is not None:
                            for item, ent in entity_mark.items():
                                # print(item, ent)
                                entity = ''
                                index, tag = item[0], item[1]
                                len_entity = len(ent)

                                for i in range(index, index + len_entity):
                                    entity = entity + raw_text[i]
                                entity_list[tag_dic[tag]] = entity
                        entity_list_ls.append(entity_list)
        else:
            raw_text, test_ids, test_masks, test_lengths = self.from_input(sentence)
            test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
            self.model.load_state_dict(torch.load(self.checkpoint, map_location={'cuda:0': 'cpu'}), False)
            self.model.eval()

            for i, dev_batch in enumerate(test_loader):
                sentence, masks, lengths = dev_batch
                batch_raw_text = raw_text[i]
                sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
                if use_cuda:
                    sentence = sentence.to(device)
                    masks = masks.to(device)

                predict_tags = self.model(sentence, masks)
                predict_tags.tolist()
                predict_tags = [i2l_dic[t.item()] for t in predict_tags[0]]
                predict_tags = predict_tags[:len(batch_raw_text)]
                pred = predict_tags[1:-1]
                raw_text = batch_raw_text[1:-1]
                entity_mark = self.split_entity_input(pred)
                entity_list = {}
                if entity_mark is not None:
                    for item, ent in entity_mark.items():
                        # print(item, ent)
                        entity = ''
                        index, tag = item[0], item[1]
                        len_entity = len(ent)

                        for i in range(index, index + len_entity):
                            entity = entity + raw_text[i]
                        entity_list[tag_dic[tag]] = entity
                # print(entity_list)
                entity_list_ls.append(entity_list)
        return entity_list_ls

    def predict_file(self, input_file, output_file):
        tag_dic = {"d": "疾病", "b": "身体", "s": "症状", "p": "医疗程序", "e": "医疗设备", "y": "药物", "k": "科室",
                   "m": "微生物类", "i": "医学检验项目"}
        raw_text, test_ids, test_masks, test_lengths = self.from_txt(input_file)
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        op_file = codecs.open(output_file, 'w', 'utf-8')
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)

            # predict_tags = self.model(sentence, masks)
            # predict_tags.tolist()
            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = [i2l_dic[t.item()] for t in predict_tags[0]]
            predict_tags = predict_tags[:len(batch_raw_text)]
            pred = predict_tags[1:-1]
            raw_text = batch_raw_text[1:-1]

            entity_mark = self.split_entity_input(pred)
            entity_list = {}
            if entity_mark is not None:
                for item, ent in entity_mark.items():
                    entity = ''
                    index, tag = item[0], item[1]
                    len_entity = len(ent)
                    for i in range(index, index + len_entity):
                        entity = entity + raw_text[i]
                    entity_list[tag_dic[tag]] = entity
            op_file.write("".join(raw_text))
            op_file.write("\n")
            op_file.write(json.dumps(entity_list, ensure_ascii=False))
            op_file.write("\n")

        op_file.close()
        print('处理完成！')
        print("结果保存至 {}".format(output_file))

    def infer_pandas_excel(self, path='export.xlsx', save_path='model_ner/result.xlsx', cmd_print=True):
        file = pd.read_excel(path)
        new_file = file.copy()
        description = file['疾病描述'].values
        symptom = []
        for des in description:
            res = self.predict_sentence(des)
            key_flag = 0
            for key, value in res.items():
                if key == "症状":
                    symptom.append(value)
                    if cmd_print:
                        print(value)
                    key_flag = 1
            if key_flag == 0:
                symptom.append(" ")
        new_file['模型预测症状（仅供参考)'] = pd.Series(symptom)
        new_file.to_excel(save_path, index=False)


if __name__ == "__main__":
    text = '''患者于3天前疑似因发热服用布洛芬出现便血，暗红色，稀烂便，1天5次，每次约200ml，中途呕血1次，鲜红色，血中有胃内容物，约300ml
    ，伴肚脐上方腹胀。患者面色苍白，有头晕、乏力、口渴，有心悸、反酸，尿量减少。无口腔溃疡、头痛、意识不清，无咳嗽、咳痰、气促，无胸闷、胸痛，无烧心、嗳气、里急后重。患者为求进一步诊治，于1
    天前来我院急诊就诊，行“血常规”，示“Hb 103g/L”，拟“消化道出血”收入我科。患者自起病以来，胃纳、精神差，睡眠可，大小便如上所述，体重体力无明显变化 '''
    # checkpoint = 'checkpoint/medical_ner/model_new_0.996.pkl'
    # """
    # [{'症状': '稀烂便'}, {'症状': '中途呕血'}, {'症状': '面色苍白'}, {'症状': '口渴'},
    # {'症状': '反酸'}, {'症状': '咳嗽'}, {'症状': '胸闷'}, {'症状': '嗳气'},
    # {'症状': '里急后重'},  {'症状': '“消化道出'}, {'症状': '精神差'}, {'症状': '睡眠可'}]
    # """

    checkpoint = 'checkpoint/medical_ner/model_old.pkl'
    """
    [{'症状': '稀烂便', '药物': '布洛芬'}, 
     {'症状': '中途呕血'}, {'身体': '胃'}, 
     {'症状': '面色苍白'}, {'症状': '口渴'}, 
     {'症状': '反酸'}, {'症状': '咳嗽'}, {'症状': '胸闷'},
      {'症状': '嗳气'}, {'症状': '里急后重'},  {'医学检验项目': '诊就诊'},
      {'疾病': '“消化道出'}, {'症状': '胃纳、精神差'},  {'医学检验项目': '体'}]
    """

    my_pred = Medical_NER_Infer(checkpoint)
    res = my_pred.predict_sentence(text)
    print(res)

    # import pandas as pd
    # path = r'1_病历.xlsx'
    # file = pd.read_excel(path, header = None).head(n=100)
    #
    # description = file[0].values
    # new_file = pd.DataFrame(description,column = ['A'])
    # symptom_ls = []
    # for des in description:
    #     res_ls = my_pred.predict_sentence(des)
    #     symptom = []
    #     key_flag = 0
    #     for res in res_ls:
    #         for key, value in res.items():
    #             try:
    #                 if key == "症状":
    #                     symptom.append(value)
    #                     print(value)
    #                     key_flag = 1
    #             except:
    #                 pass
    #         if key_flag == 0:
    #             symptom.append(" ")
    #     symptom_str = ",".join(symptom)
    #     print(symptom_str)
    #     symptom_ls.append(symptom_str)
    # new_file['模型预测症状（仅供参考)'] = pd.Series(symptom_ls)
    # new_file.to_excel('result.xlsx', index=False)
