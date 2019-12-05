from __future__ import print_function, absolute_import, division

import os
import time
import argparse

import scipy.io as sio
import torch
from sklearn.cluster import KMeans

from model import MultimodalGAN
from utils import calculate_metrics, check_dir_exist
METRIC_PRINT = 'metrics: ' + ', '.join(['{:.4f}'] * 7)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128)  # 128
parser.add_argument("--lr_g", type=float, default=1e-4,  # 1e-4
                    help="adam: learning rate for G")
parser.add_argument("--lr_d", type=float, default=1e-4,  # 1e-4
                    help="adam: learning rate for D")
parser.add_argument("--lr_ae", type=float, default=1e-4,  # 1e-4
                    help="adam: learning rate for AE")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--lamda1", type=float, default=1.0,
                    help="reg for cycle consistency")
parser.add_argument("--lamda2", type=float, default=1.0,  # 1.0
                    help="reg for reconstruction loss after cycle mapping")
parser.add_argument("--lamda3", type=float, default=1.0,  # 1.0
                    help="reg for adversarial loss")
parser.add_argument("--gan_type", type=str, default='naive',
                    choices=['naive', 'wasserstein'])
parser.add_argument("--clip_value", type=float, default=0.05,
                    help="gradient clipping")

parser.add_argument("--n_cpu", type=int, default=8,
                    help="# of cpu threads during batch generation")
parser.add_argument("--seed", type=int, default=2018)

parser.add_argument('--update_p_freq', type=int, default=10)
parser.add_argument('--update_d_freq', type=int, default=5)
parser.add_argument('--tol', type=int, default=1e-3)
parser.add_argument('--save_freq', type=int, default=25)
parser.add_argument('--log_freq', type=int, default=5)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument("--pretrain", type=str, default='None',
                    choices=['img', 'txt', 'load_ae', 'load_all', 'None'])
parser.add_argument("--dataset", type=str, default='wikipedia',
                    choices=['wikipedia', 'nuswide'])
parser.add_argument("--data_dir", type=str, default='data/wikipedia/')
parser.add_argument('--log_dir', type=str, default='log/')
parser.add_argument('--cpt_dir', type=str, default='cpt/',
                    help="dir for saved checkpoint")
parser.add_argument('--img_cptpath', type=str, default='cpt/',
                    help="path to load img AE checkpoint")
parser.add_argument('--txt_cptpath', type=str, default='cpt/',
                    help="path to load txt AE checkpoint")
parser.add_argument('--dm2c_cptpath', type=str, default='cpt/',
                    help="path to load dm2c checkpoint")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# reproducible
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.benchmark = True

METRIC_PRINT = 'metrics: ' + ', '.join(['{:.4f}'] * 7)

if __name__ == '__main__':
    config = dict()
    if args.dataset == 'wikipedia':
        config['img_input_dim'] = 2048
        config['txt_input_dim'] = 2048
        config['n_clusters'] = 10
        config['img_hiddens'] = [1024, 256, 128]
        config['txt_hiddens'] = [1024, 256, 128]
        config['img2txt_hiddens'] = [128, 256, 128]
        config['txt2img_hiddens'] = [128, 256, 128]
        # if the data include corresponding filename for each sample feature
        config['has_filename'] = True
    elif args.dataset == 'nuswide':
        config['img_input_dim'] = 1000
        config['txt_input_dim'] = 1000
        config['img_hiddens'] = [512, 256, 128]
        config['txt_hiddens'] = [512, 128]
        config['img2txt_hiddens'] = [128, 128]
        config['txt2img_hiddens'] = [128, 128]
        config['has_filename'] = False
    config['batchnorm'] = True
    config['cuda'] = use_cuda
    config['device'] = device
    current_time = time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    config['log_file'] = current_time + '.txt'

    check_dir_exist(args.log_dir)
    check_dir_exist(args.cpt_dir)
    args.cpt_dir = os.path.join(args.cpt_dir, current_time)
    os.mkdir(args.cpt_dir)

    model = MultimodalGAN(args, config)
    if use_cuda:
        model.to_cuda()

    # pretrain the autoencoders
    if args.pretrain == 'img':
        model.pretrain('img')
    elif args.pretrain == 'txt':
        model.pretrain('txt')
    elif args.pretrain == 'load_all':
        model.load_cpt(args.dm2c_cptpath)
    elif args.pretrain == 'load_ae':
        model.load_pretrain_cpt(args.img_cptpath, 'img', only_weight=True)
        model.load_pretrain_cpt(args.txt_cptpath, 'txt', only_weight=True)

        for epoch in range(args.n_epochs):
            model.train(epoch)
            train_embedding, train_target, train_modality = model.embedding(
                model.train_loader_ordered, unify_modal='img')
            test_embedding, test_target, test_modality = model.embedding(
                model.test_loader, unify_modal='img')
            kmeans = KMeans(config['n_clusters'], max_iter=1000,
                            tol=5e-5, n_init=20).fit(train_embedding)
            train_metrics = calculate_metrics(train_target, kmeans.labels_)
            y_pred = kmeans.predict(test_embedding)
            test_metrics = calculate_metrics(test_target, y_pred)
            print('>Train', METRIC_PRINT.format(*train_metrics))
            print('>Test ', METRIC_PRINT.format(*test_metrics))
            # sio.savemat('result/result_{}.mat'.format(epoch),
            #             {'X_embed_train': train_embedding,
            #              'y_pred_train': kmeans.predict(train_embedding),
            #              'y_true_train': train_target,
            #              'modal_train': train_modality,
            #              'X_embed_test': test_embedding,
            #              'y_pred_test': y_pred,
            #              'y_true_test': test_target,
            #              'modal_test': test_modality})
