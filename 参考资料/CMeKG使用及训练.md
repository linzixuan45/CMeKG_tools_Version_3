# CMeKG 工具 代码及模型


Index
---
<!-- TOC -->

- [CMeKG工具](#cmekg工具)
  - [模型下载](#模型下载)
- [依赖库](#依赖库)
- [模型使用](#模型使用)
  - [关系抽取](#医学关系抽取)
  - [医学实体识别](#医学实体识别)
  - [医学文本分词](#医学文本分词)


<!-- /TOC -->


## cmekg工具

[CMeKG网站](https://cmekg.pcl.ac.cn/)

中文医学知识图谱CMeKG
CMeKG（Chinese Medical Knowledge Graph）是利用自然语言处理与文本挖掘技术，基于大规模医学文本数据，以人机结合的方式研发的中文医学知识图谱。

CMeKG 中主要模型工具包括 医学文本分词，医学实体识别和医学关系抽取。这里是三种工具的代码、模型和使用方法。

### 模型下载

由于依赖和训练好的的模型较大，将模型放到了百度网盘中，链接如下，按需下载。

RE：链接:https://pan.baidu.com/s/1cIse6JO2H78heXu7DNewmg  密码:4s6k


NER: 链接:https://pan.baidu.com/s/16TPSMtHean3u9dJSXF9mTw  密码:shwh


分词：链接:https://pan.baidu.com/s/1bU3QoaGs2IxI34WBx7ibMQ  密码:yhek

## 依赖库

- json
- random
- numpy
- torch
- transformers
- gc
- re
- time
- tqdm

## 模型使用

### 医学关系抽取

**依赖文件**

-  pytorch_model.bin : 医学文本预训练的 BERT-base model
-  vocab.txt
-  config.json
-  model_re.pkl: 训练好的关系抽取模型文件，包含了模型参数、优化器参数等
-  predicate.json 

**使用方法**

配置参数在medical_re.py的class config里，首先在medical_re.py的class config里修改各个文件路径

- 训练

```python

from 参考代码 import medical_re

medical_re.load_schema()
medical_re.run_train()
```

model_re/train_example.json 是训练文件示例

- 使用

```python

from 参考代码 import medical_re

medical_re.load_schema()
model4s, model4po = medical_re.load_model()

text = '据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人。'  # content是输入的一段文字
res = medical_re.get_triples(text, model4s, model4po)
print(json.dumps(res, ensure_ascii=False, indent=True))
```

- 执行结果

```
[
 {
  "text": "据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、=乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人",
  "triples": [
   [
    "新冠肺炎",
    "临床表现",
    "肺炎"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "发热"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "咳嗽"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "胸闷"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "乏力"
   ],
   [
    "新冠肺炎",
    "病因",
    "自身免疫系统缺陷"
   ],
   [
    "新冠肺炎",
    "病因",
    "人传人"
   ]
  ]
 }
]
```

### 医学实体识别

调整的参数和模型在ner_constant.py中

**训练**

python3 train_ner.py


**使用示例**


medical_ner 类提供两个接口测试函数

- predict_sentence(sentence): 测试单个句子，返回:{"实体类别"：“实体”},不同实体以逗号隔开
- predict_file(input_file, output_file): 测试整个文件
文件格式每行待提取实体的句子和提取出的实体{"实体类别"：“实体”},不同实体以逗号隔开

```python
from run import medical_ner

#使用工具运行
my_pred=medical_ner()
#根据提示输入单句：“高血压病人不可食用阿莫西林等药物”
sentence=input("输入需要测试的句子:")
my_pred.predict_sentence("".join(sentence.split()))

#输入文件(测试文件，输出文件)
my_pred.predict_file("my_test.txt","outt.txt")
```

### 医学文本分词

调整的参数和模型在cws_constant.py中

**训练**

python3 train_cws.py


**使用示例**


medical_cws 类提供两个接口测试函数

- predict_sentence(sentence): 测试单个句子，返回:{"实体类别"：“实体”},不同实体以逗号隔开
- predict_file(input_file, output_file): 测试整个文件
文件格式每行待提取实体的句子和提取出的实体{"实体类别"：“实体”},不同实体以逗号隔开

```python
from run import medical_cws

#使用工具运行
my_pred=medical_cws()
#根据提示输入单句：“高血压病人不可食用阿莫西林等药物”
sentence=input("输入需要测试的句子:")
my_pred.predict_sentence("".join(sentence.split()))

#输入文件(测试文件，输出文件)
my_pred.predict_file("my_test.txt","outt.txt")
```

## 模型训练

### 参数

| 参数名      | 含义                                                         | 调整心得                                                     |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| max_length  | 单行字符长度限制                                             | 默认给150，不用动即可，BMES四位标注法单行也就一个字符        |
| batch_size  | 表示单次传递给程序用以训练的数据(样本)个数。比如我们的[训练集](https://so.csdn.net/so/search?q=训练集&spm=1001.2101.3001.7020)有`1000`个数据。这是如果我们设置`batch_size=100`，那么程序首先会用[数据集](https://so.csdn.net/so/search?q=数据集&spm=1001.2101.3001.7020)中的前`100`个参数，即第`1-100`个数据来训练模型。当训练完成后更新权重，再使用第`101-200`的个数据训练，直至第十次使用完训练集中的`1000`个数据后停止。优势：<br/>可以减少内存的使用，因为我们每次只取100个数据，因此训练时所使用的内存量会比较小。这对于我们的电脑内存不能满足一次性训练所有数据时十分有效。可以理解为训练数据集的分块训练。<br/>提高训练的速度，因为每次完成训练后我们都会更新我们的权重值使其更趋向于精确值。所以完成训练的速度较快。<br/>劣势：<br/>使用少量数据训练时可能因为数据量较少而造成训练中的梯度值较大的波动。 | 好的实验表现都是在batch size处于2~32之间得到的。当有足够算力时，选取batch size为32或更小一些。 |
| epochs      | 迭代次数，一般把数据集遍历几遍的意思                         | 目前看损失函数                                               |
| tagset_size | 根据所使用的标注法判定数量                                   | 常量表中给定好的不用动                                       |
| use_cuda    | 是否使用cuda引擎                                             | 显卡不支持，禁用                                             |

### 训练过程

1、以分词为例，首先准备好训练集、测试集、验证集，可以数据都一样，数据集需要提前进行标注，cmekg使用的是BMES四位序列标注法。

2、更改模型、模型参数、数据集的路径

3、调整训练参数，这个参数需要根据运行情况做出调整

4、开始训练

### 注意事项

1、使用标注工具对文本进行标注后的anns是BMES四位序列标注法的运行文件，由于设置的关系需要批量替换O标注的内容为S

2、标注后的结果需要将空格替换成tab字符

3、标注时可以对需要的数据才进行标注，其他不关心的数据可以省略标注

4、数据集不可以太小。