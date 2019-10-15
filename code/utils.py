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


def pickle_load(file, root_dir):
    with open(os.path.join(root_dir, file), 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        return np.array(data)


def load_data(dataset='wikipedia', root_dir='.'):
    if dataset == 'nuswide':
        train_img_feats = pickle_load('img_train_id_feats.pkl', root_dir)
        train_txt_vecs = pickle_load('train_id_bow.pkl', root_dir)
        train_labels = pickle_load('train_id_label_map.pkl', root_dir)

        test_img_feats = pickle_load('img_test_id_feats.pkl', root_dir)
        test_txt_vecs = pickle_load('test_id_bow.pkl', root_dir)
        test_labels = pickle_load('test_id_label_map.pkl', root_dir)

        train_ids = pickle_load('train_ids.pkl', root_dir)
        test_ids = pickle_load('test_ids.pkl', root_dir)
        train_labels_single = pickle_load(
            'train_id_label_single.pkl', root_dir)
        test_labels_single = pickle_load('test_id_label_single.pkl', root_dir)

        return train_img_feats, train_txt_vecs, train_labels,\
            test_img_feats, test_txt_vecs, test_labels,\
            train_ids, test_ids, train_labels_single, test_labels_single
    elif dataset == 'wikipedia':
        train_img_feats = pickle_load('train_img_feats.pkl', root_dir)
        train_txt_vecs = pickle_load('train_txt_vecs.pkl', root_dir)
        train_labels = pickle_load('train_labels.pkl', root_dir)

        test_img_feats = pickle_load('test_img_feats.pkl', root_dir)
        test_txt_vecs = pickle_load('test_txt_vecs.pkl', root_dir)
        test_labels = pickle_load('test_labels.pkl', root_dir)

        train_txt_files = pickle_load('train_txt_files.pkl', root_dir)
        train_img_files = pickle_load('train_img_files.pkl', root_dir)
        test_txt_files = pickle_load('test_txt_files.pkl', root_dir)
        test_img_files = pickle_load('test_img_files.pkl', root_dir)

        return train_img_feats, train_txt_vecs, train_labels, test_img_feats,\
            test_txt_vecs, test_labels, train_txt_files, train_img_files,\
            test_txt_files, test_img_files


def sample_single_modal(dataset='wikipedia'):
    DIR = '/home/data/dataset/cross-modal/single-label/{}/'.format(dataset)
    if dataset == 'wikipedia':
        train_img_feats, train_txt_vecs, train_labels,\
            test_img_feats, test_txt_vecs, test_labels, train_txt_files,\
            train_img_files, test_txt_files, test_img_files =\
            load_data(dataset, DIR)
        PCA_dim = 2048
        test_size = 1. / 3.
        with_file = True
    elif dataset == 'nuswide':
        train_img_feats, train_txt_vecs, train_labels,\
            test_img_feats, test_txt_vecs, test_labels,\
            train_ids, test_ids, train_labels_single, test_labels_single =\
            load_data(dataset, DIR)
        train_img_feats = np.concatenate([[train_img_feats.item()[i]] for i in train_ids], 0)
        test_img_feats = np.concatenate([[test_img_feats.item()[i]] for i in test_ids], 0)
        train_txt_vecs = np.concatenate([[train_txt_vecs.item()[i]] for i in train_ids], 0)
        test_txt_vecs = np.concatenate([[test_txt_vecs.item()[i]] for i in test_ids], 0)
        train_labels = np.concatenate([[train_labels_single.item()[i]] for i in train_ids], 0)
        test_labels = np.concatenate([[test_labels_single.item()[i]] for i in test_ids], 0)
        PCA_dim = 1000
        test_size = 1. / 4.
        with_file = False
    img_feats = np.concatenate((train_img_feats, test_img_feats), 0)
    txt_feats = np.concatenate((train_txt_vecs, test_txt_vecs), 0)
    # img_feats = PCA(n_components=PCA_dim).fit_transform(img_feats)
    # txt_feats = PCA(n_components=PCA_dim).fit_transform(txt_feats)
    labels = np.concatenate((train_labels, test_labels), 0)
    if dataset == 'wikipedia':
        img_files = np.concatenate((train_img_files, test_img_files), 0)
        txt_files = np.concatenate((train_txt_files, test_txt_files), 0)
        X = [[img_feat.squeeze(), txt_feat.squeeze(),
              img_file.squeeze(), txt_file.squeeze()]
             for img_feat, txt_feat, img_file, txt_file in zip(
            img_feats, txt_feats, img_files, txt_files)]
    elif dataset == 'nuswide':
        X = [[img_feat.squeeze(), txt_feat.squeeze()]
             for img_feat, txt_feat in zip(img_feats, txt_feats)]
    # unbalance for each class
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, stratify=labels)

    def split_modal(data, with_file=False):
        if with_file:
            data_single = [[img_feat, img_file, 0]
                           if np.random.random_integers(2) < 2
                           else [txt_feat, txt_file, 1]
                           for img_feat, txt_feat, img_file, txt_file in data]
        else:
            data_single = [[img_feat, 0]
                           if np.random.random_integers(2) < 2
                           else [txt_feat, 1]
                           for img_feat, txt_feat in data]
        data_single = np.array(data_single)
        m0, m1 = data_single[data_single[:, -1] ==
                             0], data_single[data_single[:, -1] == 1]
        return data_single, m0, m1

    X_train_single, X_train_img, X_train_txt = split_modal(X_train, with_file)
    X_test_single, X_test_img, X_test_txt = split_modal(X_test, with_file)
    img_pca = PCA(n_components=PCA_dim).fit(np.stack(X_train_img[:, 0]))
    img_feats_train = img_pca.transform(np.stack(X_train_img[:, 0]))
    idx = np.arange(len(X_train_single))[X_train_single[:, -1] == 0]
    for i in range(len(idx)):
        X_train_single[idx[i], 0] = img_feats_train[i]
        X_train_img[i, 0] = img_feats_train[i]
    img_feats_test = img_pca.transform(np.stack(X_test_img[:, 0]))
    idx = np.arange(len(X_test_single))[X_test_single[:, -1] == 0]
    for i in range(len(idx)):
        X_test_single[idx[i], 0] = img_feats_test[i]
        X_test_img[i, 0] = img_feats_test[i]
    # txt_pca = PCA(n_components=PCA_dim).fit(np.stack(X_train_txt[:, 0]))
    # txt_feats_train = txt_pca.transform(np.stack(X_train_txt[:, 0]))
    # idx = np.arange(len(X_train_single))[X_train_single[:, -1] == 1]
    # for i in range(len(idx)):
    #     X_train_single[idx[i], 0] = txt_feats_train[i]
    #     X_train_txt[i, 0] = txt_feats_train[i]
    # txt_feats_test = txt_pca.transform(np.stack(X_test_txt[:, 0]))
    # idx = np.arange(len(X_test_single))[X_test_single[:, -1] == 1]
    # for i in range(len(idx)):
    #     X_test_single[idx[i], 0] = txt_feats_test[i]
    #     X_test_txt[i, 0] = txt_feats_test[i]

    # X_single, X_img, X_txt = split_modal(X, False)
    # sio.savemat(os.path.join(DIR, 'total_file.mat'),
    #             {'X': X_single, 'y': labels})
    # sio.savemat(os.path.join(DIR, 'total_img.mat'),
    #             {'X': X_img})
    # sio.savemat(os.path.join(DIR, 'total_txt.mat'),
    #             {'X': X_txt})

    # X_train_single, X_train_img, X_train_txt = split_modal(X_train, with_file)
    # X_test_single, X_test_img, X_test_txt = split_modal(X_test, with_file)
    sio.savemat(os.path.join(DIR, 'train_file.mat'),
                {'X': X_train_single, 'y': y_train})
    sio.savemat(os.path.join(DIR, 'test_file.mat'),
                {'X': X_test_single, 'y': y_test})
    sio.savemat(os.path.join(DIR, 'train_img.mat'),
                {'X': X_train_img})
    sio.savemat(os.path.join(DIR, 'train_txt.mat'),
                {'X': X_train_txt})
# sample_single_modal(dataset='wikipedia')
# sample_single_modal(dataset='nuswide')


class MFeatDataSet(Dataset):
    '''Multimodal feature'''

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
        # modality_label = np.array([1., 0.]) if int(modality[0]) == 0 else np.array([0., 1.])

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
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
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
    return f


def get_purity(y_true, y_pred):
    c = metrics.confusion_matrix(y_true, y_pred)
    return 1. * c.max(axis=0).sum() / np.shape(y_true)[0]


def calculate_metrics(y, y_pred):
    y_new = best_map(y, y_pred)
    acc = metrics.accuracy_score(y, y_new)
    ar = get_ar(y, y_pred)
    nmi = get_nmi(y, y_pred)
    f = get_fpr(y, y_pred)
    purity = get_purity(y, y_pred)
    return acc, ar, nmi, f, purity
