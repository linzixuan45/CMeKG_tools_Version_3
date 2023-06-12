from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
from tools.utils import load_vocab, load_data, recover_label, get_ner_fmeasure

from All_models.model_ner.bert_lstm_crf import BERT_LSTM_CRF
from All_models.Optimizer import Lion
import os
import time
import glob

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


class Trainer:
    def __init__(self, checkpoint_in='/root/医学实体识别_version_1/model_ner/medical_ner/model.pkl',
                 save_dir='model_ner/train/model/'):
        self.checkpoint = checkpoint_in
        self.medical_bert = 'checkpoint/Bert_embedding'
        self.save_model_dir = save_dir
        self.max_length = 450
        self.batch_size = 2
        self.epochs = 100
        self.tagset_size = len(l2i_dic)
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
        self.model = BERT_LSTM_CRF(self.medical_bert, self.tagset_size, 768, 200, 2,
                                   dropout_ratio=0.5, dropout1=0.5, use_cuda=self.use_cuda)
        if self.checkpoint:
            self.log(f'Restoring from checkpoint: {self.checkpoint}')
            self.log(self.model.load_state_dict(
                torch.load(self.checkpoint, map_location={'cuda:0': 'cpu'}), False))
        if use_cuda:
            self.model = self.model.to(self.device)
        self.optimizer = Lion(self.model.parameters(), lr=0.0001, weight_decay=0.00001)

    def init_dataset(self):

        """step_1  all data"""
        # train_file = glob.glob('dataset/data_train_test_dev/train_*')
        # test_file = glob.glob('dataset/data_train_test_dev/test_*')
        # dev_file = glob.glob('dataset/data_train_test_dev/dev_*')
        # """step_2  finturning"""
        # train_file = glob.glob('dataset/data_finturning/train_*')
        # test_file = glob.glob('dataset/data_finturning/test_*')
        # dev_file = glob.glob('dataset/data_finturning/dev_*')
        # """step2  finturning  Row  0.685"""
        # train_file = glob.glob('dataset/data_finturning/train_Row*')
        # test_file = glob.glob('dataset/data_finturning/test_Row*')
        # dev_file = glob.glob('dataset/data_finturning/dev_Row*')

        train_Row = glob.glob('dataset/data_train_test_dev/train_Row*')
        test_Row = glob.glob('dataset/data_train_test_dev/test_Row*')
        dev_Row = glob.glob('dataset/data_train_test_dev/dev_Row*')

        train_CMeEE = glob.glob('dataset/data_train_test_dev/train_CMeEE*')
        test_CMeEE = glob.glob('dataset/data_train_test_dev/test_CMeEE*')
        dev_CMeEE = glob.glob('dataset/data_train_test_dev/dev_CMeEE*')

        train_CMedCausal = glob.glob('dataset/data_train_test_dev/train_CMedCausal*')
        test_CMedCausal = glob.glob('dataset/data_train_test_dev/train_CMedCausal*')
        dev_CMedCausal = glob.glob('dataset/data_train_test_dev/train_CMedCausal*')
        # """Row CMeEE CMedCausal"""
        # train_file = train_Row+train_CMeEE+train_CMedCausal
        # test_file = test_Row+test_CMeEE+test_CMedCausal
        # dev_file = dev_Row+dev_CMeEE+dev_CMedCausal
        """ CMeEE CMedCausal """
        train_file = train_CMeEE + train_CMedCausal
        test_file = test_CMeEE + test_CMedCausal
        dev_file = dev_CMeEE + dev_CMedCausal
        """Row  0.685"""

        vocab_file = 'checkpoint/Bert_embeading/vocab.txt'
        vocab = load_vocab(vocab_file)
        self.vocab_reverse = {v: k for k, v in vocab.items()}
        print('max_length', self.max_length)
        train_data = load_data(train_file, max_length=self.max_length, label_dic=l2i_dic, vocab=vocab, glob_dir_ls=True)
        train_ids = torch.LongTensor([temp.input_id for temp in train_data])
        train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
        train_tags = torch.LongTensor([temp.label_id for temp in train_data])
        train_lenghts = torch.LongTensor([temp.lenght for temp in train_data])
        train_dataset = TensorDataset(train_ids, train_masks, train_tags, train_lenghts)
        self.train_loader = DataLoader(train_dataset, shuffle=False, batch_size=self.batch_size)
        dev_data = load_data(dev_file, max_length=self.max_length, label_dic=l2i_dic, vocab=vocab, glob_dir_ls=True)
        dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
        dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
        dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])
        dev_lenghts = torch.LongTensor([temp.lenght for temp in dev_data])
        dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags, dev_lenghts)
        self.dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=self.batch_size)
        test_data = load_data(test_file, max_length=self.max_length, label_dic=l2i_dic, vocab=vocab, glob_dir_ls=True)
        test_ids = torch.LongTensor([temp.input_id for temp in test_data])
        test_masks = torch.LongTensor([temp.input_mask for temp in test_data])
        test_tags = torch.LongTensor([temp.label_id for temp in test_data])
        test_lenghts = torch.LongTensor([temp.lenght for temp in test_data])
        test_dataset = TensorDataset(test_ids, test_masks, test_tags, test_lenghts)
        self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size)

    def evaluate(self, flag='dev'):
        self.model.eval()
        pred = []
        gold = []
        print('evaluate')
        if flag == 'dev':
            loader = self.dev_loader
            print(" this is dev evaluate")
        elif flag == 'test':
            loader = self.test_loader
            print(" this is test evaluate")
        with torch.no_grad():
            for i, dev_batch in enumerate(tqdm(loader)):
                sentence, masks, tags, lengths = dev_batch
                sentence, masks, tags, lengths = Variable(sentence), Variable(masks), Variable(tags), Variable(lengths)
                if self.use_cuda:
                    sentence = sentence.to(self.device)
                    masks = masks.to(self.device)
                    tags = tags.to(self.device)
                predict_tags = self.model(sentence, masks)
                loss = self.model.neg_log_likelihood_loss(sentence, masks, tags)
                pred.extend([t for t in predict_tags.tolist()])
                gold.extend([t for t in tags.tolist()])
            pred_label, gold_label = recover_label(pred, gold, l2i_dic, i2l_dic)
            print('dev loss {}'.format(loss.item()))
            pred_label_1 = [t[1:] for t in pred_label]
            gold_label_1 = [t[1:] for t in gold_label]
            acc, p, r, f = get_ner_fmeasure(gold_label_1, pred_label_1)
            print('acc:{} p: {}，r: {}, f: {}'.format(acc, p, r, f))
            return p, r, f

    def train(self):
        best_f = -100
        for epoch in range(self.epochs):
            print('epoch: {}，train'.format(epoch))
            for i, train_batch in enumerate(tqdm(self.train_loader)):
                sentence, masks, tags, lengths = train_batch
                sentence, masks, tags, lengths = Variable(sentence), Variable(masks), Variable(tags), Variable(lengths)
                if self.use_cuda:
                    sentence = sentence.to(self.device)
                    masks = masks.to(self.device)
                    tags = tags.to(self.device)
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.model.neg_log_likelihood_loss(sentence, masks, tags)
                loss.backward()
                self.optimizer.step()
            print('epoch: {}，train loss: {}'.format(epoch, loss.item()))
            self.evaluate(flag='dev')
            p, r, f = self.evaluate(flag='test')
            if f > best_f:
                print('参数保存开始保存')
                best_f = f
                model_name = self.save_model_dir + "/" + 'new' + str(float('%.3f' % best_f)) + ".pkl"
                torch.save(self.model.state_dict(), model_name)
                print('保存成功')


if __name__ == "__main__":
    # checkpoint = "/root/医学实体识别_version_1/checkpoint/CMeEE/new0.914.pkl"
    # # checkpoint = '/root/医学实体识别_version_1/checkpoint/CMedCausal/new0.913.pkl'
    # # checkpoint = '/root/医学实体识别_version_1/checkpoint/CMeEE/new0.909.pkl'
    # checkpoint = '/root/医学实体识别_version_1/model_ner/medical_ner/model.pkl'
    # checkpoint = '/root/医学实体识别_version_1/checkpoint/ALL/new0.713.pkl'
    checkpoint = 'checkpoint/medical_ner/model_old.pkl'
    save_dir = "checkpoint/CMeEE_CMedCausal"
    os.makedirs(save_dir, exist_ok=True)
    train_ner = Trainer(checkpoint, save_dir)
