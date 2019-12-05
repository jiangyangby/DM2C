from __future__ import print_function, absolute_import, division

import os
import pickle
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics.cluster.supervised import contingency_matrix
from munkres import Munkres


class MFeatDataSet(Dataset):
    '''Mixed-modal feature'''

    def __init__(self, file_mat, has_filename=False):
        self.file_mat = sio.loadmat(file_mat)
        self.lens = len(self.file_mat['X'])
        self.has_filename = has_filename

    def __getitem__(self, index):
        if self.has_filename:
            feat, file, modality = self.file_mat['X'][index]
        else:
            feat, modality = self.file_mat['X'][index]
        feat = feat.squeeze().astype(np.float32)
        cluster_label = self.file_mat['y'][0][index]
        cluster_label = np.float32(cluster_label) - 1
        modality_label = np.float32(modality[0])

        return np.float32(index), feat, modality_label, cluster_label

    def __len__(self):
        return self.lens


class SFeatDataSet(Dataset):
    '''Single modal feature'''

    def __init__(self, file_mat):
        self.file_mat = sio.loadmat(file_mat)
        self.lens = len(self.file_mat['X'])

    def __getitem__(self, index):
        feat = self.file_mat['X'][index][0].squeeze().astype(np.float32)
        return feat

    def __len__(self):
        return self.lens


def best_map(L1, L2):
    # L1 should be the ground-truth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = (L1 == Label1[i]).astype(float)
        for j in range(nClass2):
            ind_cla2 = (L2 == Label2[j]).astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def get_ar(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


def get_nmi(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')


def get_fpr(y_true, y_pred):
    n_samples = np.shape(y_true)[0]
    c = contingency_matrix(y_true, y_pred, sparse=True)
    tk = np.dot(c.data, np.transpose(c.data)) - n_samples  # TP
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples  # TP+FP
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples  # TP+FN
    precision = 1. * tk / pk if tk != 0. else 0.
    recall = 1. * tk / qk if tk != 0. else 0.
    f = 2 * precision * recall / (precision + recall) if (precision +
                                                          recall) != 0. else 0.
    return f, precision, recall


def get_purity(y_true, y_pred):
    c = metrics.confusion_matrix(y_true, y_pred)
    return 1. * c.max(axis=0).sum() / np.shape(y_true)[0]


def calculate_metrics(y, y_pred):
    y_new = best_map(y, y_pred)
    acc = metrics.accuracy_score(y, y_new)
    ar = get_ar(y, y_pred)
    nmi = get_nmi(y, y_pred)
    f, p, r = get_fpr(y, y_pred)
    purity = get_purity(y, y_pred)
    return acc, ar, nmi, f, p, r, purity


def check_dir_exist(dir_):
    if not os.path.isdir(dir_):
        os.mkdir(dir_)
