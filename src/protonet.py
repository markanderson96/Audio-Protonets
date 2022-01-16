import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig

from datagenerator import *
from features import *
from utils import euclidean_dist, man_dist, norm_dist

class Protonet(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        def conv_block(in_channels, out_channels, kernel_size=3, pooling=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, padding_mode='zeros'),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.MaxPool2d(pooling)
        )
        self.conf = conf
        self.encoder = nn.Sequential(
            conv_block(1, conf.set.n_dim),
            conv_block(conf.set.n_dim, conf.set.n_dim),
            conv_block(conf.set.n_dim, conf.set.n_dim),
            conv_block(conf.set.n_dim, conf.set.n_dim),
            conv_block(conf.set.n_dim, conf.set.n_dim),
            conv_block(conf.set.n_dim, conf.set.n_dim, kernel_size=(3,2)),
        )
        self.example_input_array = torch.rand(1, 125, conf.features.n_mels)

        self.emb_counter = 0
        self.data_counter = 0
        self.y_val_emb = torch.tensor([])
        self.y_out_emb = torch.tensor([])
        self.distance = conf.train.distance
        self.norm = conf.train.norm
    
    def forward(self, x):
        (num_samples, seq_len, fft_bins) = x.shape
        x = x.view(-1, 1, seq_len, fft_bins)
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def loss_function(self, Y_in, Y_target):
        def support_idxs(c):
            return Y_target.eq(c).nonzero()[:n_support].squeeze(1)
        
        n_support = self.conf.train.n_shot

        Y_target = Y_target.to('cpu')
        Y_in = Y_in.to('cpu')

        classes = torch.unique(Y_target)
        n_classes = len(classes)
        p = n_classes * n_support

        n_query = Y_target.eq(classes[0].item()).sum().item() - n_support
        s_idxs = list(map(support_idxs, classes))
        prototypes = torch.stack([Y_in[idx].mean(0) for idx in s_idxs])

        q_idxs = torch.stack(list(map(lambda c:Y_target.eq(c).nonzero()[n_support:], classes))).view(-1)
        q_samples = Y_in.cpu()[q_idxs]
        
        if self.distance == 'euclidean':
            dists_e = euclidean_dist(q_samples, prototypes)
            if self.norm:
                dists_n = norm_dist(q_samples, prototypes)
                dists = torch.sqrt(dists_e + 0.5 * dists_n)
            else:
                dists = dists_e
        elif self.distance == 'manhattan':
            dists_m = man_dist(q_samples, prototypes)
            if self.norm:
                dists_n = norm_dist(q_samples, prototypes)
                dists = torch.sqrt(dists_m + 0.5 * dists_n)
            else:
                dists = dists_m
        
        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

        target_idxs = torch.arange(0, n_classes)
        target_idxs = target_idxs.view(n_classes, 1, 1)
        target_idxs = target_idxs.expand(n_classes, n_query, 1).long()
        loss_val = -log_p_y.gather(2, target_idxs).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_idxs.squeeze()).float().mean()

        return loss_val, acc_val

    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y_out = self(X)
        train_loss, train_acc = self.loss_function(Y_out, Y)

        self.log('train_loss', train_loss)
        self.log('train_acc', train_acc)
        
        log = {'train_acc':train_acc}
        self.log_dict(log, prog_bar=True)
        self.logger.experiment.add_scalars("losses", {"train_loss": train_loss}, global_step=self.current_epoch)

        return {'loss':train_loss}

    def validation_step(self, batch, batch_idx):
        X, Y = batch         
        Y_out = self(X)
        val_loss, val_acc = self.loss_function(Y_out, Y)

        # get data for t-SNE every 10 steps
        if self.emb_counter % 5 == 0:
            if self.data_counter % 20 == 0:
                self.y_val_emb = torch.cat((self.y_val_emb, Y.detach().cpu()))
                self.y_out_emb = torch.cat((self.y_out_emb, Y_out.detach().cpu()))
            self.data_counter += 1

        #log = {'val_loss':val_loss, 'val_acc':val_acc}
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)
        self.logger.experiment.add_scalars("losses", {"val_loss": val_loss}, global_step=self.current_epoch)

        return {'val_loss':val_loss, 'val_acc':val_acc}
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        log = {'avg_val_loss':val_loss, 'avg_val_acc':val_acc}
        #self.log('avg_val_loss', val_loss)
        #self.log('avg_val_acc', val_acc)

        if self.emb_counter % 5 == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_embedding(self.y_out_emb,
                                      metadata=self.y_val_emb,
                                      global_step=self.current_epoch,
                        )
        
            self.emb_counter = 0
            self.data_counter = 0

        self.emb_counter += 1
        self.y_val_emb = torch.tensor([])
        self.y_out_emb = torch.tensor([])

        return {'log':log}
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 
                                    lr=self.conf.train.lr, 
                                    momentum=self.conf.train.momentum)

        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer,
                                                       factor=self.conf.train.factor,
                                                       patience=self.conf.train.patience,
                                                       verbose=True),
                        'monitor': 'val_loss'
        }

        return {'optimizer':optimizer, 'lr_scheduler': lr_scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def predict_step(self, batch, batch_idx, dataloader_idx):
        X, Y = batch         
        Y_out = self(X)
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        
