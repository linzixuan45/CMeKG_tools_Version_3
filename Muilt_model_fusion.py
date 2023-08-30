#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/6/29 0:18
# @Author  : Linxuan Jiang
# @File    : Muilt_model_fusion.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT

import re
import pandas as pd
from Medical_ner_Inference import Medical_NER_Infer
from Medical_re_Inference import RE_Inference


class FusionModel:
    def __init__(self, min_max_text_limit=(7, 15)):
        self.drop_ls = pd.read_csv('Test_711/drop_ls.csv',header=None).values.reshape(-1).tolist()
        self.drop_ls = list(set(self.drop_ls))
        checkpoint_row = 'checkpoint/Pretrain_weight/medical_ner/model.pkl'
        checkpoint_new = 'checkpoint/Pretrain_weight/medical_ner/new0.996.pkl'
        self.Ner_Long = Medical_NER_Infer(checkpoint_row)  # 长文本症状提取模型
        self.Ner_Short = Medical_NER_Infer(checkpoint_new)  # 短文本症状提取模型
        self.Medical_Re = RE_Inference()  # 症状补充提取模型
        self.min_max_text_limit = min_max_text_limit

        EXCEPT_PUNCTUATION = "-、\——.。,，， ""'( ) ！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"""
        self.EXCEPT_PUNCTUATION = "[{}]+".format(EXCEPT_PUNCTUATION)


    @staticmethod
    def get_symptom(ner_word):
        """
        ner_word: list
        """
        result_ls = []
        for temp_dict in ner_word:
            for key, value in temp_dict.items():
                if key in ['症状']:
                    result_ls.append(value)
        result_ls = list(set(result_ls))
        return result_ls

    @staticmethod
    def get_triple_symptom(triple_dict_ls):
        result_ls = []
        for triple_dict in triple_dict_ls:
            for key, value in triple_dict.items():
                if key in ['triples']:
                    for triple in value:
                        if triple[1] in ['临床表现']:
                            result_ls.append(triple[0])
                            result_ls.append(triple[2])
        result_ls = list(set(result_ls))
        return result_ls

    def predict(self, text):
        """
        return list
        """

        ner_word_row = self.Ner_Long.predict_sentence(text, split_limit=self.min_max_text_limit[1])  # 捕捉长文本
        ner_word_new = self.Ner_Short.predict_sentence(text, split_limit=self.min_max_text_limit[0])
        triple_dict_ls = self.Medical_Re.predict_triples(text)

        symptom_row = self.get_symptom(ner_word_row)
        symptom_new = self.get_symptom(ner_word_new)
        symptom_re = self.get_triple_symptom(triple_dict_ls)

        fusion_symptom = list(set(symptom_re + symptom_new + symptom_row))
        #fusion_symptom = list(set(symptom_re + symptom_row))

        final_symptom = []
        for word in fusion_symptom:
            temp_word = re.sub(self.EXCEPT_PUNCTUATION, "", word)
            if len(temp_word)!=0: # 字符串存在
                if temp_word[0] in ['(', ')', ',', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                    temp_word = temp_word[1:]
                try:
                    if temp_word[-1] in ['(', ')', ',', '.', '。']:
                        temp_word = temp_word[:-1]
                except:
                    pass
                if len(temp_word) > 1:  # 长度小于等于1的症状直接丢弃
                    if len(temp_word) > 6:  # 这个值是做异常判断，一般症状长度小于9，大于9的再次预测
                        ner_word = self.Ner_Short.predict_sentence(temp_word, split_limit=5)
                        symptom_new = self.get_symptom(ner_word)
                        for temp_word in symptom_new:
                            if temp_word not in self.drop_ls and len(temp_word)>1:
                                final_symptom.append(temp_word)
                        continue

                    if temp_word not in self.drop_ls:
                        final_symptom.append(temp_word)
                else:
                    pass

        final_symptom = list(set(final_symptom))
        return final_symptom


if __name__ == "__main__":

    # 参数意义: 短文本症状提取模型最大短文本 7，  长文本症状提取模型最大长文本 15

    """
    要求1： 不允许出现疾病名称
    要求2：输出字符大于2个，且不允许出现标点符号
    
    1.只要excel的关于症状的干净的数据 带标点符号的   
    不是症状的，不需要给  
    
    2. 模型只能输出症状  
    一个列表里面只能出现症状 
    烈性肠道传染病是疾病 不是症状
    """
    row_pd = pd.read_excel("Test_711/原始数据.xlsx")[:10]
    file_pd = pd.DataFrame()
    index=0
    fusion_model = FusionModel(min_max_text_limit=(5, 6))
    for i, row in enumerate(row_pd.values):

        symptom_ls = fusion_model.predict(row[2])
        print(row[2])
        print(symptom_ls)
        for symptom in symptom_ls:
            file_pd.loc[index, '疾病编码'] = row[0]
            file_pd.loc[index, '疾病名称'] = row[1]
            file_pd.loc[index, '疾病症状'] = symptom
            file_pd.loc[index, '描述'] = row[2]
            index +=1


    file_pd.to_excel('result.xlsx')
