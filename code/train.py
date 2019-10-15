from __future__ import print_function, absolute_import, division

import argparse

import scipy.io as sio
import torch
from sklearn.cluster import KMeans

from model_cycle2 import MultimodalGAN
from utils import calculate_metrics
METRIC_PRINT = 'metrics: ' + ', '.join(['{:.4f}'] * 5)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128)  # 128
parser.add_argument("--lr_g", type=float, default=1e-4,  # 1e-4
                    help="adam: learning rate for G")
parser.add_argument("--lr_d", type=float, default=1e-4,  # 1e-4
                    help="adam: learning rate for D")
parser.add_argument("--lr_c", type=float, default=1e-4,
                    help="adam: learning rate for clustering layer")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--lamda1", type=float, default=1.0,
                    help="reg for txt reconstruction")
parser.add_argument("--lamda2", type=float, default=1.0,  # 10.0
                    help="reg for cluster loss")
parser.add_argument("--lamda3", type=float, default=1.0,  # 1.0
                    help="reg for adversarial loss")
parser.add_argument("--lamda4", type=float, default=10,  # 100
                    help="reg for centroid dissimilarity")
parser.add_argument("--lamda5", type=float, default=1.0,  # 100
                    help="reg for gradient penalty")
parser.add_argument("--gan_type", type=str, default='naive',
                    choices=['naive', 'wasserstein', 'wasserstein-div'])
parser.add_argument("--clip_value", type=float, default=0.05,
                    help="gradient clipping")

parser.add_argument("--n_cpu", type=int, default=8,
                    help="# of cpu threads during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10,
                    help="number of classes for dataset")

parser.add_argument('--update_p_freq', type=int, default=10)
parser.add_argument('--update_d_freq', type=int, default=5)
parser.add_argument('--tol', type=int, default=1e-3)
parser.add_argument('--save_freq', type=int, default=25)
parser.add_argument('--log_freq', type=int, default=5)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument("--pretrain", type=str, default='None',
                    choices=['img', 'txt', 'None'])
parser.add_argument("--dataset", type=str, default='wikipedia',
                    choices=['wikipedia', 'nuswide'])
parser.add_argument("--data_dir", type=str,
                    default='/home/data/dataset/cross-modal/single-label/'
                    'wikipedia/')
parser.add_argument('--log_dir', type=str, default='log/')
parser.add_argument('--cpt_dir', type=str, default='model/')
parser.add_argument('--res_dir', type=str, default='result/')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True

METRIC_PRINT = 'metrics: ' + ', '.join(['{:.4f}'] * 7)

if __name__ == '__main__':
    config = dict()
    config['img_input_dim'] = 2048
    config['txt_input_dim'] = 2048
    config['batchnorm'] = True
    config['cuda'] = use_cuda
    config['device'] = device
    config['n_clusters'] = 10
    config['img_hiddens'] = [1024, 256, 128]
    config['txt_hiddens'] = [1024, 256, 128]
    config['img2txt_hiddens'] = [128, 256, 128]
    config['txt2img_hiddens'] = [128, 256, 128]
    config['train_prefix'] = 'train'  # total, train
    config['test_prefix'] = 'test'  # total, test
    config['has_filename'] = True
    model = MultimodalGAN(args, config)
    if use_cuda:
        model.to_cuda()

    # pretrain the autoencoders
    if args.pretrain == 'img':
        model.pretrain('img')  # 5e-4
    elif args.pretrain == 'txt':
        model.pretrain('txt')  # 2e-5

    # train AEs and discriminator (without clustering layer)
    model.set_logger()
    model.load_pretrain_cpt('model/{}_img_pretrain_checkpt_149.pkl'.format(args.dataset),
                            'img', only_weight=True)
    model.load_pretrain_cpt('model/{}_txt_pretrain_checkpt_139.pkl'.format(args.dataset),
                            'txt', only_weight=True)  # 139
    # model.load_cpt('/home/data/wikipedia_checkpt_169_2720.pkl')
    for epoch in range(args.n_epochs):
        model.train(epoch)
        train_embedding, train_target = model.embedding_pred(
            model.train_loader_ordered, specific=False)
        test_embedding, test_target = model.embedding_pred(
            model.test_loader, specific=False)
        kmeans = KMeans(config['n_clusters'], max_iter=1000,
                        tol=5e-5, n_init=20).fit(train_embedding)
        train_metrics = calculate_metrics(train_target, kmeans.labels_)
        y_pred = kmeans.predict(test_embedding)
        test_metrics = calculate_metrics(test_target, y_pred)
        print('>Train', METRIC_PRINT.format(*train_metrics))
        print('>Test ', METRIC_PRINT.format(*test_metrics))

    model.close_logger()
