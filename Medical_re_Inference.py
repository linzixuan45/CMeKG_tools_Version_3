#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/6/9 11:47
# @Author  : Linxuan Jiang
# @File    : Medical_re_Inference.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright (MIT) 2023 - 2023 Linxuan Jiang, Inc. All Rights Reserved

import torch
import json
import re
import numpy as np
from All_models.model_re.medical_re import Model4s, Model4po
from transformers import BertTokenizer
import math



class config:
    batch_size = 32
    max_seq_len = 256
    num_p = 23
    learning_rate = 1e-5
    EPOCH = 2
    PATH_TRAIN = 'model_re/train/data/train_data.json'
    PATH_BERT = "checkpoint/Pretrain_weight/Bert_embedding"
    PATH_MODEL = "checkpoint/Pretrain_weight/medical_re/model_re.pkl"
    PATH_SAVE = 'checkpoint/Pretrain_weight/medical_re'
    tokenizer = BertTokenizer.from_pretrained("checkpoint/Pretrain_weight/Bert_embedding/vocab.txt")

    id2predicate = {0: '相关疾病', 1: '相关症状', 2: '临床表现', 3: '检查', 4: '用法用量', 5: '贮藏',
                    6: '性状', 7: '禁忌', 8: '不良反应', 9: '有效期', 10: '注意事项', 11: '适应症',
                    12: '成份', 13: '病因', 14: '治疗', 15: '规格', 16: '并发症', 17: '药物相互作用',
                    18: '主治', 19: '入药部位', 20: '功效', 21: '性味', 22: '功能主治'}

    predicate2id = {'相关疾病': 0, '相关症状': 1, '临床表现': 2, '检查': 3, '用法用量': 4, '贮藏': 5,
                    '性状': 6, '禁忌': 7, '不良反应': 8, '有效期': 9, '注意事项': 10, '适应症': 11,
                    '成份': 12, '病因': 13, '治疗': 14, '规格': 15, '并发症': 16, '药物相互作用': 17,
                    '主治': 18, '入药部位': 19, '功效': 20, '性味': 21, '功能主治': 22}


def find_subtext_index(text, keywords):
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


class RE_Inference:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_model(config.PATH_MODEL)


    def init_model(self, checkpoint):
        self.model4s = Model4s(config)
        self.model4po = Model4po(config)
        if checkpoint:
            wight = torch.load(checkpoint, map_location='cpu')
            self.model4s.load_state_dict(wight['model4s_state_dict'], False)
            self.model4po.load_state_dict(wight['model4po_state_dict'], False)

        self.model4po.to(self.device)
        self.model4s.to(self.device)

    def get_triples(self, content):
        if len(content) == 0:
            return []
        text_list = content.split('。')[:-1]
        res = []
        for text in text_list:
            if len(text) > 254:
                text = text[:254]
            triples = self.extract_spoes(text)
            res.append({
                'text': text,
                'triples': triples
            })
        return res

    def predict_triples(self, sentence):
        split_limit = 128
        res = []
        if sentence == '':
            print("输入为空！请重新输入")
            return
        if len(sentence) > split_limit:
            juhao_ls = find_subtext_index(sentence, ['。', '，', "、", ',', ])
            juhao_ls = [value[1] for value in juhao_ls]
            split_pices = math.ceil(len(sentence) / split_limit)
            split_bias = [0]
            for j in range(split_pices + 1):  # 找到切分的下标也就是偏置
                for i, value in enumerate(juhao_ls):
                    if i + 1 < len(juhao_ls) and juhao_ls[i] < split_limit * j < juhao_ls[i + 1]:
                        split_bias.append(value)  # value 是需要切分的位置
            split_bias.append(len(sentence))
            split_bias = list(set(split_bias))
            split_bias = sorted(split_bias)
            for i, vlaue in enumerate(split_bias):
                if i + 1 < len(split_bias):
                    temp_text = sentence[split_bias[i]: split_bias[i + 1]]
                    temp_text = "".join(map(str, temp_text))
                    triples = self.extract_spoes(temp_text)
                    res.append({
                        'text': temp_text,
                        'triples': triples
                    })
        else:
            triples = self.extract_spoes(sentence)
            res.append({
                'text': sentence,
                'triples': triples
            })
        return res

    def extract_spoes(self, text):
        """
        要求1： 预测的主语 和谓语必须要在同一句话中
        要求2： 谓语，关系只有23类，是一个分类任务
        return: a list of many tuple of (s, p, o)
        """
        # 处理text
        with torch.no_grad():

            """字按照词表编码为向量， 编码过程 """
            tokenizer = config.tokenizer # 21128个字符的词表
            max_seq_len = config.max_seq_len  # 256
            token_ids = torch.tensor(
                tokenizer.encode(text, max_length=max_seq_len, pad_to_max_length=True, add_special_tokens=True)).view(1, -1)
            # token_ids [1, 256]

            if len(text) > max_seq_len - 2:
                text = text[:max_seq_len - 2]

            # mask_id 给padding 添加mask 预测的结果丢弃
            mask_ids = torch.tensor([1] * (len(text) + 2) + [0] * (max_seq_len - len(text) - 2)).view(1, -1)
            segment_ids = torch.tensor([0] * max_seq_len).view(1, -1)
            token_ids, mask_ids,segment_ids = token_ids.to(self.device), mask_ids.to(self.device),segment_ids.to(self.device)

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
            subject_ids = torch.tensor(subjects).view(1, -1).to(self.device)

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

        """向量按照词表还原为字， 解码过程"""
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
    text = '''现病史:本次就诊主要疾病的首次发作情况，包括它的诱因，性质，阵发性或持续性，程度，与进食或体位的关系，是否影响活动，伴随症状等，缓解方式等。有意义的阴性体征，用药(他人叙述的疾病或药物名称均应加引号)，效果。(时间不能用英文缩写来代替，必须用文字表述。) 做过何种检查及结果。患者自发病以来，神志、精神、睡眠、食欲、大小便、体重改变、体力变化。
既往史:平素身体健康状况，否认结核、肝炎、疟疾等传染病及传染病密切接触史，否认“高血压”病史(时间、治疗及控制情况)、否认“糖尿病”病史，否认“心脏病”病史，否认“脑血管疾病”病史，否认输血史，无献血史，无手术外伤史(何时、何地及治疗情况)，否认药物过敏史(过敏表现，缓解方式)，否认食物过敏史。预防接种史随当地计划免疫。
系统回顾:
呼吸系统:无慢性咳嗽、咯痰、咯血史，无呼吸困难，无发热、盗汗，无结核患者密切接触史循环系统: 无心悸、气促、发维，无心前区疼痛，无高血压史，无晕厥、水肿病史，无动脉硬化，无风湿热病史。
消化系统:无腹痛、腹胀、反酸、暖气，无呕血、便血，无食欲不振、恶心或呕吐史，大便正常。
泌尿生殖系统:无尿频、尿急、尿痛，无腰痛及排尿困难，无眼脸浮肿，无血尿，无肾毒性药物应用造血系统: 无苍白、乏力等，皮肤黏膜无瘀点、紧寝，无反复鼻出血或牙龈出血。
内分泌系统及代谢:无畏寒、多汗、无头痛或视力障碍，无食欲异常、烦渴、多尿等，毛发分布均匀，第二性征无改
神经精神系统: 无头痛、失眠、嗜睡，无喷射性呕吐、记忆力改变，无意识障碍、瘫痪、昏厥、痉挛，无视力障碍、感觉及运动异常，无性格改变
肌肉骨骼系统:无关节肿痛，无运动障碍，无肢体麻木，无痉挛萎缩或瘫痪史。
个人史:生长于原籍，文化程度，无外地长期居住史(地点及居住年限》，无疫区、疫水接触史，无工业粉尘、毒物、放射性物质接触史，无牧区、矿山、高氮区、低碘区居住史，平日生活规律，否认吸毒史、否认吸烟史，(多少年，每日多少支，戒烟多年) 否认饮酒史， (饮酒多年，每日饮酒折合酒精约多少g)否认冶游史。
婚姻史:已婚，22岁结婚，配偶健康，育有1子
月经生育史: 月经来潮，-------，月经量中等，无凝血块，无痛经。 (中度痛经，是否影响日常活动) 孕一产---，6年前足月妊娠因“胎膜早破”剖宫产1男活婴，无流产、死产、难产，育有----子女
家族史:父亲体健， (管患疾病及治愈情况;去世原因，年龄)母亲体健，1兄体健，1子体健，家族无遗传病病史。 '''
    # text = r'据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人。'
    Intity = RE_Inference(config)

    # print(Intity.get_triples(text))

    print(Intity.predict_triples(text))

    """ [{'text': '据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人',
      'triples': [('新冠肺炎', '临床表现', '肺炎'), ('新冠肺炎', '临床表现', '发热'), ('新冠肺炎', '临床表现', '咳嗽'),
                  ('新冠肺炎', '临床表现', '胸闷'), ('新冠肺炎', '临床表现', '乏力'),
                  ('新冠肺炎', '病因', '自身免疫系统缺陷'), ('新冠肺炎', '病因', '人传人')]}]"""
