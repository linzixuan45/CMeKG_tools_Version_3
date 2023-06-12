# coding: utf-8
import pandas as pd


class InputFeatures(object):
    def __init__(self, text, label, input_id, label_id, input_mask, length):
        self.text = text
        self.label = label
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask
        self.lenght = length


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


def load_file(file_path):
    contents = open(file_path, encoding='utf-8').readlines()
    text = []
    label = []
    texts = []
    labels = []
    for line in contents:
        if line != '\n':
            line = line.strip().split('\t')
            text.append(line[0])
            label.append(line[-1])
        else:
            texts.append(text)
            labels.append(label)
            text = []
            label = []
    return texts, labels




def load_data(file_path, max_length, label_dic, vocab, load_pd_file=True, glob_dir_ls = False):

    # 返回InputFeatures的list
    if load_pd_file:
        if glob_dir_ls:
            pd_file = None
            for i, path in enumerate(file_path):
                if i == 0:
                    pd_file = pd.read_csv(path)
                else:
                    temp_file = pd.read_csv(path)
                    pd_file = pd.concat([pd_file, temp_file], axis=0)
            pd_file.reset_index(drop=True, inplace=True)
            texts, labels = pd_file['text'], pd_file['label_BMEO']
        else:
            file = pd.read_csv(file_path)
            texts, labels = file['text'], file['label_BMEO']
    else:
        texts, labels = load_file(file_path)
    assert len(texts) == len(labels)
    result = []
    for i in range(len(texts)):
        if type(labels[i]) != str:
            label_str = labels[i].values[0]
            text_str = texts[i].values[0]
        else:
            label_str = labels[i]
            text_str = texts[i]
        text, label = list(text_str), label_str.split('\t')
        # print(len(text), len(label))
        assert len(text) == len(label)
        token = text
        label = label
        if len(token) > max_length - 2:  # 截断
            token = token[0:(max_length - 2)]
            label = label[0:(max_length - 2)]

        tokens_f = ['[CLS]'] + token + ['[SEP]']

        label_f = ["<start>"] + label + ['<eos>']
        #  我 vocab-》 2770    bert-》 【768】 embeading 向量

        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        label_ids = [label_dic[i] for i in label_f]
        input_mask = [1] * len(input_ids)

        length = [len(tokens_f)]

        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length
        feature = InputFeatures(text=tokens_f, label=label_f, input_id=input_ids, input_mask=input_mask,
                                label_id=label_ids, length=length)
        result.append(feature)

    return result


def recover_label(pred_var, gold_var, l2i_dic, i2l_dic):
    assert len(pred_var) == len(gold_var)
    pred_variable = []
    gold_variable = []
    for i in range(len(gold_var)):
        start_index = gold_var[i].index(l2i_dic['<start>'])
        end_index = gold_var[i].index(l2i_dic['<eos>'])
        pred_variable.append(pred_var[i][start_index:end_index])
        gold_variable.append(gold_var[i][start_index:end_index])

    pred_label = []
    gold_label = []
    for j in range(len(gold_variable)):
        pred_label.append([i2l_dic[t] for t in pred_variable[j]])
        gold_label.append([i2l_dic[t] for t in gold_variable[j]])

    return pred_label, gold_label


class SegmenterEvaluation():
    def evaluate(self, original_labels, predict_labels):
        right, predict = self.get_order(original_labels, predict_labels)
        print('right, predict: ', right, predict)
        right_count = self.rightCount(right, predict)
        if right_count == 0:
            recall = 0
            precision = 0
            f1 = 0
            error = 1
        else:
            recall = right_count / len(right)
            precision = right_count / len(predict)
            f1 = (2 * recall * precision) / (precision + recall)
            error = (len(predict) - right_count) / len(right)
        return precision, recall, f1, error, right, predict

    def rightCount(self, rightList, predictList):
        count = set(rightList) & set(predictList)
        return len(count)

    def get_order(self, original_labels, predict_labels):

        assert len(original_labels) == len(predict_labels)

        start = 1
        end = len(original_labels) - 1  # 当 len(original_labels) -1 > 1的时候,只要有一个字就没问题
        # 按照origin的长度，且删去开头结尾符
        original_labels = original_labels[start:end]
        predict_labels = predict_labels[start:end]

        def merge(labelList):
            # 输入标签字符串，返回一个个词的(begin,end+1)元组
            new_label = []
            chars = ""
            for i, label in enumerate(labelList):
                if label not in ("B", "M", "E", "S"):  # 可能是其他标签
                    if len(chars) != 0:
                        new_label.append(chars)
                    new_label.append(label)
                    chars = ""
                elif label == "B":
                    if len(chars) != 0:
                        new_label.append(chars)
                    chars = "B"
                elif label == "M":
                    chars += "M"
                elif label == "S":
                    if len(chars) != 0:
                        new_label.append(chars)
                    new_label.append("S")
                    chars = ""
                else:
                    new_label.append(chars + "E")
                    chars = ""
            if len(chars) != 0:
                new_label.append(chars)
            orderList = []
            start = 0
            end = 0
            for each in new_label:
                end = start + len(each)
                orderList.append((start, end))
                start = end

            return orderList

        right = merge(original_labels)
        predict = merge(predict_labels)
        return right, predict


def get_f1(gold_label, pred_label):
    assert len(gold_label) == len(pred_label)
    sege = SegmenterEvaluation()
    total_right = 0
    total_pred = 0
    total_gold = 0
    for i in range(len(gold_label)):
        temp_gold, temp_predict = sege.get_order(gold_label[i], pred_label[i])
        temp_right = sege.rightCount(temp_gold, temp_predict)
        total_right += temp_right
        total_gold += len(temp_gold)
        total_pred += len(temp_predict)
    recall = total_right / total_gold
    precision = total_right / total_pred
    # edit by cpc
    f1 = (2 * recall * precision) / (precision + recall + float("1e-8"))
    return precision, recall, f1


def save_model(path, model, epoch):
    pass


def load_model(path, model):
    return model


def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1  # 预测正确标签数
        all_tag += len(golden_list)  # 所有真正标签数
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)  # 句子中的存在实体[（start,end）类型]
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))  # 真实标签和预测标签位置是否正确
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num  # 准确率：正确预测实体数/所有预测实体数
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num  # 召回率：正确预测实体数/真实标签正确实体数量
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag  # 准确率：正确标签/所有标签
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    return accuracy, precision, recall, f_measure


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


if __name__ == "__main__":
    pass
