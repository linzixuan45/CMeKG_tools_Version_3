#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023/6/14 13:17
# @Author  : Linxuan Jiang
# @File    : SVD_Cos.py
# @IDE     : PyCharm
# @Email   : 1195860834@qq.com
# Copyright MIT
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import torch


def compute_kernel_bias(vecs, n_components=256):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu

#
# # 定义两个张量
# torch.manual_seed(42)
# a = torch.randn((3, 512))
# b = torch.randn((3, 512))

import torch
import torch.nn.functional as F
import numpy as np
import scipy
