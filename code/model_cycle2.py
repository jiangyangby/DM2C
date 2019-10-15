# python3 train.py --lr_g 1e-4 --lr_d 5e-5 --gan_type wasserstein --n_epochs 250 --weight_decay 1e-4 --lamda3 0.5 --lamda1 1
# python3 train.py --lr_g 1e-4 --lr_d 5e-5 --gan_type wasserstein --n_epochs 250 --weight_decay 0 --lamda3 0.5 --lamda1 1 --lr_c 5e-4
from __future__ import print_function, absolute_import, division

import logging
import time
import os
import itertools
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans

from utils import MFeatDataSet, SFeatDataSet, calculate_metrics

logging.basicConfig(level=logging.INFO,
                    filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s: %(name)s [%(levelname)s] %(message)s')
info_string1 = ('Epoch: %3d/%3d|Batch: %2d/%2d||D_loss: %.4f|D1_loss: %.4f|'
                'D2_loss: %.4f||G_loss: %.4f|R1_loss: %.4f|R2_loss: %.4f|R121_loss: %.4f|'
                'R212_loss: %.4f|R121r_loss: %.4f|R212r_loss: %.4f')


def calc_grad_norm(x, y):
    p = 6
    grad_out = autograd.Variable(torch.cuda.FloatTensor(x.size(
        0), 1).fill_(1.0), requires_grad=False)
    grad = autograd.grad(
        y, x, grad_out,
        create_graph=True, retain_graph=True,
        only_inputs=True, allow_unused=True)[0]
    grad_norm = grad.view(
        grad.size(0), -1).pow(2).sum(1) ** (p / 2)
    return grad_norm


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    n = min(real_samples.size(0), fake_samples.size(0))
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((n, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples[:n] + ((1 - alpha)
                                                * fake_samples[:n])).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = autograd.Variable(torch.cuda.FloatTensor(
        n, 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class DeepAE(nn.Module):
    """DeepAE: FC AutoEncoder"""

    def __init__(self, input_dim=1, hiddens=[1], batchnorm=False):
        super(DeepAE, self).__init__()
        self.depth = len(hiddens)
        self.channels = [input_dim] + hiddens  # [5, 3, 3]

        encoder_layers = []
        for i in range(self.depth):
            encoder_layers.append(
                nn.Linear(self.channels[i], self.channels[i + 1]))
            if i < self.depth - 1:
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                if batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(self.channels[i + 1]))
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for i in range(self.depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self.channels[i], self.channels[i - 1]))
            decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if i > 1 and batchnorm:
                decoder_layers.append(nn.BatchNorm1d(self.channels[i - 1]))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent


class MultimodalGAN:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.logger = logging.getLogger(__name__)
        assert config['img_hiddens'][-1] == config['txt_hiddens'][-1],\
            'Inconsistent latent dim!'

        self._build_dataloader()

        # Generator
        self.latent_dim = config['img_hiddens'][-1]
        self.imgAE = DeepAE(input_dim=config['img_input_dim'],
                            hiddens=config['img_hiddens'],
                            batchnorm=config['batchnorm'])
        self.txtAE = DeepAE(input_dim=config['txt_input_dim'],
                            hiddens=config['txt_hiddens'],
                            batchnorm=config['batchnorm'])
        self.img2txt = DeepAE(input_dim=self.latent_dim,
                              hiddens=config['img2txt_hiddens'],
                              batchnorm=config['batchnorm'])
        self.txt2img = DeepAE(input_dim=self.latent_dim,
                              hiddens=config['txt2img_hiddens'],
                              batchnorm=config['batchnorm'])

        # Discriminator
        self.D_img = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim / 4)),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(int(self.latent_dim / 2), int(self.latent_dim / 4)),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.latent_dim / 4), 1)
        )  # modal classifier
        self.D_txt = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim / 4)),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(int(self.latent_dim / 2), int(self.latent_dim / 4)),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(self.latent_dim / 4), 1)
        )  # modal classifier

        # Optimizer
        self.optimizer_G1 = optim.Adam(
            self.imgAE.parameters(), lr=self.args.lr_g,
            betas=(self.args.b1, self.args.b2),
            weight_decay=self.args.weight_decay)
        self.optimizer_G2 = optim.Adam(
            self.txtAE.parameters(), lr=self.args.lr_g,
            betas=(self.args.b1, self.args.b2),
            weight_decay=self.args.weight_decay)
        params = [{'params': itertools.chain(
            self.imgAE.parameters(), self.txtAE.parameters())},
            {'params': itertools.chain(
                self.img2txt.parameters(), self.txt2img.parameters()),
             'lr': self.args.lr_c}]
        if self.args.gan_type == 'wasserstein':
            self.optimizer_D = optim.RMSprop(
                itertools.chain(
                    self.D_img.parameters(), self.D_txt.parameters()),
                lr=self.args.lr_d,
                weight_decay=self.args.weight_decay)
            self.optimizer_G = optim.RMSprop(
                params,
                lr=self.args.lr_g,
                weight_decay=self.args.weight_decay)
        else:
            self.optimizer_D = optim.Adam(
                itertools.chain(
                    self.D_img.parameters(), self.D_txt.parameters()),
                lr=self.args.lr_d,
                betas=(self.args.b1, self.args.b2),
                weight_decay=self.args.weight_decay)
            self.optimizer_G = optim.Adam(
                params,
                lr=self.args.lr_g, betas=(self.args.b1, self.args.b2),
                weight_decay=self.args.weight_decay)

        self.set_writer()
        self.adv_loss_fn = F.binary_cross_entropy_with_logits

    def pretrain(self, modal='img'):
        self.set_model_status(training=True)
        if modal == 'img':
            AE = self.imgAE
            optimizer = self.optimizer_G1
        elif modal == 'txt':
            AE = self.txtAE
            optimizer = self.optimizer_G2
        dataloader = self._build_pretrain_dataloader(modal)
        for epoch in range(self.args.n_epochs):
            for i, feats in enumerate(dataloader):
                feats = feats.cuda()
                optimizer.zero_grad()

                feats_recon, feats_latent = AE(feats)
                recon_loss = F.mse_loss(feats, feats_recon)

                recon_loss.backward()
                optimizer.step()

                if (i + 1) % self.args.log_freq == 0:
                    time_string = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                    print(time_string,
                          "Epoch: %d/%d|Batch: %d/%d|Recon_loss: %.4f"
                          % (epoch, self.args.n_epochs, i,
                             len(dataloader), recon_loss.item())
                          )

            if (epoch + 1) % self.args.save_freq == 0:
                self.save_pretrain_cpt(epoch, modal)

    def train(self, epoch):
        self.set_model_status(training=True)
        for step, (ids, feats, modalitys, labels) in enumerate(self.train_loader):
            batch_size = feats.shape[0]
            ids, feats, modalitys, labels =\
                ids.cuda(), feats.cuda(), modalitys.cuda(), labels.cuda()

            modalitys = modalitys.view(-1)

            img_idx = modalitys == 0
            txt_idx = modalitys == 1

            # -----------------
            #  Train Generator without Clustering Layer
            # -----------------

            self.optimizer_G.zero_grad()

            img_feats = feats[img_idx]
            txt_feats = feats[txt_idx]
            img_batch_size = img_feats.size(0)
            txt_batch_size = txt_feats.size(0)
            imgs_recon, imgs_latent = self.imgAE(img_feats)
            txts_recon, txts_latent = self.txtAE(txt_feats)
            img2txt_recon, _ = self.img2txt(imgs_latent)
            img_latent_recon, _ = self.txt2img(img2txt_recon)
            txt2img_recon, _ = self.txt2img(txts_latent)
            txt_latent_recon, _ = self.img2txt(txt2img_recon)

            img_recon_loss = F.mse_loss(img_feats, imgs_recon)
            txt_recon_loss = F.mse_loss(txt_feats, txts_recon)
            img_cycle_loss = F.l1_loss(imgs_latent, img_latent_recon)
            txt_cycle_loss = F.l1_loss(txts_latent, txt_latent_recon)
            img_cycle_recon_loss = F.mse_loss(
                img_feats, self.imgAE.decoder(img_latent_recon))
            txt_cycle_recon_loss = F.mse_loss(
                txt_feats, self.txtAE.decoder(txt_latent_recon))
            recon_loss = img_recon_loss + txt_recon_loss +\
                (img_cycle_loss + txt_cycle_loss) * self.args.lamda1 +\
                (img_cycle_recon_loss + txt_cycle_recon_loss) * self.args.lamda2

            img_real = torch.ones(img_batch_size, 1).cuda()
            img_fake = torch.zeros(img_batch_size, 1).cuda()
            txt_real = torch.ones(txt_batch_size, 1).cuda()
            txt_fake = torch.zeros(txt_batch_size, 1).cuda()

            if self.args.gan_type == 'naive':
                d_loss = self.adv_loss_fn(self.D_img(txt2img_recon), txt_real) +\
                    self.adv_loss_fn(self.D_txt(img2txt_recon), img_real)
            elif 'wasserstein' in self.args.gan_type:
                d_loss = -self.D_img(txt2img_recon).mean() - \
                    self.D_txt(img2txt_recon).mean()

            G_loss = recon_loss + self.args.lamda3 * d_loss

            G_loss.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if (step + 1) % self.args.update_d_freq == 0:
                self.optimizer_D.zero_grad()

                if self.args.gan_type == 'naive':
                    img_D_loss = (self.adv_loss_fn(self.D_img(imgs_latent.detach()), img_real) +
                                  self.adv_loss_fn(self.D_img(txt2img_recon.detach()), txt_fake)) / 2
                    txt_D_loss = (self.adv_loss_fn(self.D_txt(txts_latent.detach()), txt_real) +
                                  self.adv_loss_fn(self.D_txt(img2txt_recon.detach()), img_fake)) / 2
                    D_loss = (img_D_loss + txt_D_loss) * self.args.lamda3
                elif self.args.gan_type == 'wasserstein':
                    img_D_loss = self.D_img(txt2img_recon.detach()).mean() -\
                        self.D_img(imgs_latent.detach()).mean()
                    txt_D_loss = self.D_txt(img2txt_recon.detach()).mean() -\
                        self.D_txt(txts_latent.detach()).mean()
                    D_loss = (img_D_loss + txt_D_loss) * self.args.lamda3
                elif self.args.gan_type == 'wasserstein-gp':
                    img_gp = compute_gradient_penalty(
                        self.D_img, imgs_latent.data, txt2img_recon.data)
                    txt_gp = compute_gradient_penalty(
                        self.D_txt, txts_latent.data, img2txt_recon.data)
                    img_D_loss = self.D_img(txt2img_recon.detach()).mean() -\
                        self.D_img(imgs_latent.detach()).mean() + \
                        self.args.lamda5 * img_gp
                    txt_D_loss = self.D_txt(img2txt_recon.detach()).mean() -\
                        self.D_txt(txts_latent.detach()).mean() + \
                        self.args.lamda5 * txt_gp
                    D_loss = (img_D_loss + txt_D_loss) * self.args.lamda3
                D_loss.backward()
                self.optimizer_D.step()

                if self.args.gan_type == 'wasserstein':
                    for p in self.D_img.parameters():
                        p.data.clamp_(-self.args.clip_value,
                                      self.args.clip_value)
                    for p in self.D_txt.parameters():
                        p.data.clamp_(-self.args.clip_value,
                                      self.args.clip_value)

            if (step + 1) % self.args.log_freq == 0:
                time_string = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                print(time_string, info_string1 % (
                    epoch, self.args.n_epochs, step, len(self.train_loader),
                    D_loss.item(), img_D_loss.item(), txt_D_loss.item(),
                    G_loss.item(), img_recon_loss.item(),
                    txt_recon_loss.item(), img_cycle_loss.item(),
                    txt_cycle_loss.item(), img_cycle_recon_loss.item(),
                    txt_cycle_recon_loss.item()))
                self.writer.add_scalar(
                    'Train/G_loss', G_loss.item(),
                    step + len(self.train_loader) * epoch)
                self.writer.add_scalar(
                    'Train/D_loss', D_loss.item(),
                    step + len(self.train_loader) * epoch)

        if epoch > 10 and (epoch + 1) % self.args.save_freq == 0:
            self.save_cpt(epoch)

    def embedding(self, dataloader):
        self.set_model_status(training=False)
        with torch.no_grad():
            latent = None
            for step, (ids, feats, modalitys, labels) in enumerate(dataloader):
                batch_size = feats.shape[0]
                feats, modalitys = feats.cuda(), modalitys.cuda()
                img_idx = modalitys.view(-1) == 0
                txt_idx = modalitys.view(-1) == 1
                imgs_recon, imgs_latent = self.imgAE(feats[img_idx])
                txts_recon, txts_latent = self.txtAE(feats[txt_idx])
                latent_code = torch.zeros(batch_size, self.latent_dim).cuda()
                latent_code[img_idx] = imgs_latent
                latent_code[txt_idx] = txts_latent
                latent = latent_code if step == 0 else torch.cat(
                    [latent, latent_code], 0)
            return latent

    def embedding_pred(self, dataloader, modal='img', specific=False):
        self.set_model_status(training=False)
        with torch.no_grad():
            latent = None
            target = None
            for step, (ids, feats, modalitys, labels) in enumerate(dataloader):
                batch_size = feats.shape[0]
                feats, modalitys = feats.cuda(), modalitys.cuda()
                img_idx = modalitys.view(-1) == 0
                txt_idx = modalitys.view(-1) == 1
                imgs_recon, imgs_latent = self.imgAE(feats[img_idx])
                txts_recon, txts_latent = self.txtAE(feats[txt_idx])
                latent_code = torch.zeros(batch_size, self.latent_dim).cuda()
                if not specific:
                    if modal == 'img':
                        txt2img_recon, _ = self.txt2img(txts_latent)
                        latent_code[img_idx] = imgs_latent
                        latent_code[txt_idx] = txt2img_recon
                    else:
                        img2txt_recon, _ = self.img2txt(imgs_latent)
                        latent_code[img_idx] = img2txt_recon
                        latent_code[txt_idx] = txts_latent
                else:
                    latent_code[img_idx] = imgs_latent
                    latent_code[txt_idx] = txts_latent
                latent = latent_code if step == 0 else torch.cat(
                    [latent, latent_code], 0)
                target = labels if step == 0 else torch.cat(
                    [target, labels], 0)
            return latent.cpu().numpy(), target.cpu().numpy()

    def _build_dataloader(self):
        kwargs = {'num_workers': self.args.n_cpu, 'pin_memory': True}
        train_data = MFeatDataSet(
            file_mat=os.path.join(self.args.data_dir, '{}_file.mat'.format(
                self.config['train_prefix'])),
            has_filename=self.config['has_filename'])
        self.train_loader = DataLoader(dataset=train_data,
                                       batch_size=self.args.batch_size,
                                       shuffle=True, **kwargs)
        self.train_loader_ordered = DataLoader(dataset=train_data,
                                               batch_size=self.args.batch_size,
                                               shuffle=False, **kwargs)

        test_data = MFeatDataSet(
            file_mat=os.path.join(self.args.data_dir, '{}_file.mat'.format(
                self.config['test_prefix'])),
            has_filename=self.config['has_filename'])
        self.test_loader = DataLoader(dataset=test_data,
                                      batch_size=self.args.batch_size,
                                      shuffle=False, **kwargs)

    def _build_pretrain_dataloader(self, modal='img'):
        kwargs = {'num_workers': self.args.n_cpu, 'pin_memory': True}
        train_modal_data = SFeatDataSet(
            file_mat=os.path.join(self.args.data_dir,
                                  '{}_{}.mat'.format(
                                      self.config['train_prefix'], modal)))
        train_modal_loader = DataLoader(dataset=train_modal_data,
                                        batch_size=self.args.batch_size,
                                        shuffle=True, **kwargs)
        return train_modal_loader

    def set_model_status(self, training=True):
        if training:
            self.imgAE.train()
            self.txtAE.train()
            self.img2txt.train()
            self.txt2img.train()
            self.D_img.train()
            self.D_txt.train()
        else:
            self.imgAE.eval()
            self.txtAE.eval()
            self.img2txt.eval()
            self.txt2img.eval()
            self.D_img.eval()
            self.D_txt.eval()

    def to_cuda(self):
        self.imgAE.cuda()
        self.txtAE.cuda()
        self.img2txt.cuda()
        self.txt2img.cuda()
        self.D_img.cuda()
        self.D_txt.cuda()

    def save_cpt(self, epoch):
        state_dict = {'epoch': epoch,
                      'G1_state_dict': self.imgAE.state_dict(),
                      'G2_state_dict': self.txtAE.state_dict(),
                      'G12_state_dict': self.img2txt.state_dict(),
                      'G21_state_dict': self.txt2img.state_dict(),
                      'D1_state_dict': self.D_img.state_dict(),
                      'D2_state_dict': self.D_txt.state_dict(),
                      'optimizer_G': self.optimizer_G.state_dict(),
                      'optimizer_D': self.optimizer_D.state_dict()
                      }
        cptname = '{}_checkpt_{}.pkl'.format(self.args.dataset, epoch)
        cptpath = os.path.join(self.args.cpt_dir, cptname)
        print("> Save checkpoint '{}'".format(cptpath))
        torch.save(state_dict, cptpath)

    def save_pretrain_cpt(self, epoch, modal='img'):
        if modal == 'img':
            AE = self.imgAE
            optimizer = self.optimizer_G1
        elif modal == 'txt':
            AE = self.txtAE
            optimizer = self.optimizer_G2
        state_dict = {'epoch': epoch,
                      'AE_state_dict': AE.state_dict(),
                      'optimizer': optimizer.state_dict()}
        cptname = '{}_{}_pretrain_checkpt_{}.pkl'.format(
            self.args.dataset, modal, epoch)
        cptpath = os.path.join(self.args.cpt_dir, cptname)
        print("> Save checkpoint '{}'".format(cptpath))
        torch.save(state_dict, cptpath)

    def load_cpt(self, cptpath):
        if os.path.isfile(cptpath):
            print("> Load checkpoint '{}'".format(cptpath))
            dicts = torch.load(cptpath)
            self.epoch = dicts['epoch']
            self.imgAE.load_state_dict(dicts['G1_state_dict'])
            self.txtAE.load_state_dict(dicts['G2_state_dict'])
            self.img2txt.load_state_dict(dicts['G12_state_dict'])
            self.txt2img.load_state_dict(dicts['G21_state_dict'])
            self.D_img.load_state_dict(dicts['D1_state_dict'])
            self.D_txt.load_state_dict(dicts['D2_state_dict'])
            self.optimizer_G.load_state_dict(dicts['optimizer_G'])
            self.optimizer_D.load_state_dict(dicts['optimizer_D'])
            # self.scheduler.load_state_dict(dicts['scheduler'])
        else:
            print("> No checkpoint found at '{}'".format(cptpath))

    def load_pretrain_cpt(self, cptpath, modal='img', only_weight=False):
        if os.path.isfile(cptpath):
            print("> Load checkpoint '{}'".format(cptpath))
            if modal == 'img':
                AE = self.imgAE
                optimizer = self.optimizer_G1
            elif modal == 'txt':
                AE = self.txtAE
                optimizer = self.optimizer_G2
            dicts = torch.load(cptpath)
            AE.load_state_dict(dicts['AE_state_dict'])
            if not only_weight:
                self.epoch = dicts['epoch']
                optimizer.load_state_dict(dicts['optimizer'])
        else:
            print("> No checkpoint found at '{}'".format(cptpath))

    def set_writer(self):
        print('> Create writer at \'{}\''.format(self.args.log_dir))
        self.writer = SummaryWriter(self.args.log_dir)

    def set_logger(self):
        time_string = time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        self.logger = open(os.path.join(
            self.args.log_dir, time_string + '.txt'), 'w')

    def close_logger(self):
        self.logger.close()
