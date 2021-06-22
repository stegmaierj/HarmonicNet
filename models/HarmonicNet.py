# -*- coding: utf-8 -*-
"""
# HarmonicNet.
# Copyright (C) 2021 D. Eschweiler, M. Rethwisch, S.Koppers, J. Stegmaier
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the Liceense at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Please refer to the documentation for more information about the software
# as well as for installation instructions.
#
# If you use this application for your work, please cite the following 
# publication:
#
# D. Eschweiler, M. Rethwisch, S.Koppers, J. Stegmaier, 
# "Spherical Harmonics for Shape-Constrained 3D Cell Segmentation", ISBI, 2021.
#
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import json

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataloader.harmonic_dataloader import MeristemH5Dataset
from ThirdParty.radam import RAdam


class HarmonicNet_module(nn.Module):
    """Implementation of the 3D U-Net architecture.
    """

    def __init__(self, in_channels, coefficients, feat_channels=16, norm_method='instance', **kwargs):
        super(HarmonicNet_module, self).__init__()
        
        self.in_channels = in_channels
        self.coefficients = coefficients
        self.feat_channels = feat_channels
        self.norm_method = norm_method # instance | batch | none
        
        if self.norm_method == 'instance':
            self.norm = nn.InstanceNorm3d
        elif self.norm_method == 'batch':
            self.norm = nn.BatchNorm3d
        elif self.norm_method == 'none':
            self.norm = nn.Identity
        else:
            raise ValueError('Unknown normalization method "{0}". Choose from "instance|batch|none".'.format(self.norm_method))
        
        
        # Define layer instances       
        self.conv_in = nn.Sequential(
            nn.Conv3d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            self.norm(feat_channels)
            )
          
        self.conv_pre = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            self.norm(feat_channels)
            )
        
        
        self.down1 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels*4, kernel_size=4, padding=1, stride=2),
            nn.PReLU(feat_channels*4),
            self.norm(feat_channels*4)
            )
        self.down1_conv = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            self.norm(feat_channels*4)
            )
        
        self.down2 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*8, kernel_size=4, padding=1, stride=2),
            nn.PReLU(feat_channels*8),
            self.norm(feat_channels*8)
            )
        self.down2_conv = nn.Sequential(
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            self.norm(feat_channels*8)
            )
        
        self.down3 = nn.Sequential(
            nn.Conv3d(feat_channels*8, feat_channels*16, kernel_size=4, padding=1, stride=2),
            nn.PReLU(feat_channels*16),
            self.norm(feat_channels*16)
            )
        self.down3_conv = nn.Sequential(
            nn.Conv3d(feat_channels*16, feat_channels*16, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*16),
            nn.Conv3d(feat_channels*16, feat_channels*16, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*16),
            self.norm(feat_channels*16)
            )
        
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(feat_channels*16, feat_channels*8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.PReLU(feat_channels*8),
            self.norm(feat_channels*8)
            )
        self.up1_conv = nn.Sequential(
            nn.Conv3d(feat_channels*16, feat_channels*8, kernel_size=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            self.norm(feat_channels*8)
            )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(feat_channels*8, feat_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.PReLU(feat_channels*4),
            self.norm(feat_channels*4)
            )
        self.up2_conv = nn.Sequential(
            nn.Conv3d(feat_channels*8, feat_channels*4, kernel_size=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            self.norm(feat_channels*4)
            )
            
    
        self.det1 = nn.Sequential(
            nn.Conv3d(feat_channels*16, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, 1, kernel_size=1),
            nn.Sigmoid()
            )        
        self.shape1 = nn.Sequential(
            nn.Conv3d(feat_channels*16, feat_channels*16, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*16),
            nn.Conv3d(feat_channels*16, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, coefficients, kernel_size=1)
            )
        
        self.det2 = nn.Sequential(
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, 1, kernel_size=1),
            nn.Sigmoid()
            )        
        self.shape2 = nn.Sequential(
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, coefficients, kernel_size=1)
            )
        
        self.det3 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, 1, kernel_size=1),
            nn.Sigmoid()
            )        
        self.shape3 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*4),
            nn.Conv3d(feat_channels*4, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels*8),
            nn.Conv3d(feat_channels*8, coefficients, kernel_size=1)
            )
    
    
    def forward(self, img):
        
        conv_pre1 = self.conv_in(img)
        conv_pre2 = self.conv_pre(conv_pre1)
        conv_pre3 = self.conv_pre(conv_pre2+conv_pre1)
        conv_pre4 = self.conv_pre(conv_pre3+conv_pre2)
        conv_pre5 = self.conv_pre(conv_pre4+conv_pre3)
        
        down1 = self.down1(conv_pre5)
        down1_conv = self.down1_conv(down1)
        
        down2 = self.down2(down1_conv+down1)
        down2_conv = self.down2_conv(down2)
        
        down3 = self.down3(down2_conv+down2)
        down3_conv = self.down3_conv(down3)
        
        up1 = self.up1(down3_conv+down3)
        up1_conv = self.up1_conv(torch.cat((up1,down2_conv),1))
        
        up2 = self.up2(up1_conv+up1)
        up2_conv = self.up2_conv(torch.cat((up2,down1_conv),1))
        
        det1 = self.det1(down3_conv)
        shape1 = self.shape1(down3_conv)
        out1 = torch.cat((det1,shape1),1)
        
        det2 = self.det2(up1_conv)
        shape2 = self.shape2(up1_conv)
        out2 = torch.cat((det2,shape2),1)
        
        det3 = self.det3(up2_conv)
        shape3 = self.shape3(up2_conv)
        out3 = torch.cat((det3,shape3),1)
        
        return out1, out2, out3
        
    
    
    
class HarmonicNet(pl.LightningModule):
    
    def __init__(self, hparams):
        super(HarmonicNet, self).__init__()
        
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams
        self.augmentation_dict = {}

        # get the number of coefficients
        self.num_coefficients = int((self.hparams.sh_order+1)**2)
        
        # networks
        self.network = HarmonicNet_module(in_channels=hparams.in_channels, coefficients=self.num_coefficients, feat_channels=hparams.feat_channels, norm_method=hparams.norm_method)

        # cache for generated images
        self.last_predictions = None
        self.last_imgs = None
        self.last_masks = None
        


    def forward(self, z):
        return self.network(z)
    
    
    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        
        # Load the state dict
        state_dict = torch.load(pretrained_file)['state_dict']
        
        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)
            
        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())
        
        layers = []
        for layer in param_dict:
            if strict and not 'network.'+layer in state_dict:
                if verbose:
                    print('Could not find weights for layer "{0}"'.format(layer))
                continue
            try:
                param_dict[layer].data.copy_(state_dict['network.'+layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print('Error at layer {0}:\n{1}'.format(layer, e))
        
        self.network.load_state_dict(param_dict)
        
        if verbose:
            print('Loaded weights for the following layers:\n{0}'.format(layers))
        

    def loss_centroid(self, y_hat, y):
        loss = F.l1_loss(y_hat, y, reduction='none')
        weight = y*torch.clamp(torch.sum(y<0.5),1,y.numel()) + (1-y)*torch.clamp(torch.sum(y>0.5),1,y.numel())
        weight = torch.div(weight, y.numel())
        loss = torch.mul(loss, weight)
        loss = torch.sum(loss)
        loss = torch.div(loss, torch.clamp(torch.sum(weight), 1, y.numel()))
        return loss
    
    def loss_encoding(self, y_hat, y, mask):
        loss = F.l1_loss(y_hat, y, reduction='none')
        loss = torch.mul(loss, mask) 
        loss = torch.sum(loss)
        loss = torch.div(loss, torch.clamp(torch.sum(mask)*self.num_coefficients, self.num_coefficients, y.numel()))
        return loss

    def training_step(self, batch, batch_idx):
        
        # Get image ans mask of current batch
        self.last_imgs = batch['image']
        
        self.centroid_small = batch['stride{0}/centroid_map'.format(self.hparams.strides[2])]
        self.encoding_small = batch['stride{0}/encoding_map'.format(self.hparams.strides[2])]
        
        self.centroid_medium = batch['stride{0}/centroid_map'.format(self.hparams.strides[1])]
        self.encoding_medium = batch['stride{0}/encoding_map'.format(self.hparams.strides[1])]
                                      
        self.centroid_large = batch['stride{0}/centroid_map'.format(self.hparams.strides[0])]
        self.encoding_large = batch['stride{0}/encoding_map'.format(self.hparams.strides[0])]
        
        # generate images
        self.pred_small, self.pred_medium, self.pred_large = self.forward(self.last_imgs)
                
        # get the centroid losses
        loss_centroid_small = self.loss_centroid(self.pred_small[:,0:1,...], self.centroid_small)
        loss_centroid_medium = self.loss_centroid(self.pred_medium[:,0:1,...], self.centroid_medium)
        loss_centroid_large = self.loss_centroid(self.pred_large[:,0:1,...], self.centroid_large)
        loss_centroid = (1/6 * loss_centroid_small + 2/6 * loss_centroid_medium + 3/6 * loss_centroid_large) 
        
        # get the encoding losses
        loss_encoding_small = self.loss_encoding(self.pred_small[:,1:,...], self.encoding_small, self.centroid_small)
        loss_encoding_medium = self.loss_encoding(self.pred_medium[:,1:,...], self.encoding_medium, self.centroid_medium)
        loss_encoding_large = self.loss_encoding(self.pred_large[:,1:,...], self.encoding_large, self.centroid_large)
        loss_encoding = (1/6 * loss_encoding_small + 2/6 * loss_encoding_medium + 3/6 * loss_encoding_large)
        
        loss = loss_centroid * self.hparams.centroid_weight + \
               loss_encoding/self.num_coefficients * self.hparams.encoding_weight
        tqdm_dict = {'centroid_loss': loss_centroid, 'encoding_loss':loss_encoding, 'epoch': self.current_epoch}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        
        if torch.isnan(loss):
            print('Image minmax: {0}/{1}'.format(self.last_imgs.min(), self.last_imgs.max()))
            print('Cent_small:{0}, Cent_medium:{1}, Cent_large:{2}, Enc_small:{3}, Enc_medium:{4}, Enc_large:{5}'.format(self.centroid_small.min(),\
                                                                                                                         self.centroid_medium.min(),\
                                                                                                                         self.centroid_large.min(),\
                                                                                                                         self.encoding_small.min(),\
                                                                                                                         self.encoding_medium.min(),\
                                                                                                                         self.encoding_large.min()))
            print('Cent_small:{0}, Cent_medium:{1}, Cent_large:{2}, Enc_small:{3}, Enc_medium:{4}, Enc_large:{5}'.format(loss_centroid_small,\
                                                                                                                         loss_centroid_medium,\
                                                                                                                         loss_centroid_large,\
                                                                                                                         loss_encoding_small,\
                                                                                                                         loss_encoding_medium,\
                                                                                                                         loss_encoding_large))
        
        return output
        
    
    def test_step(self, batch, batch_idx):
        test_imgs = batch['image']
        
        centroid_small = batch['stride{0}/centroid_map'.format(self.hparams.strides[2])]
        encoding_small = batch['stride{0}/encoding_map'.format(self.hparams.strides[2])]
        
        centroid_medium = batch['stride{0}/centroid_map'.format(self.hparams.strides[1])]
        encoding_medium = batch['stride{0}/encoding_map'.format(self.hparams.strides[1])]
                                      
        centroid_large = batch['stride{0}/centroid_map'.format(self.hparams.strides[0])]
        encoding_large = batch['stride{0}/encoding_map'.format(self.hparams.strides[0])]
        
        pred_small, pred_medium, pred_large = self.forward(test_imgs)
        
        loss_centroid_small = self.loss_centroid(pred_small[:,0:1,...], centroid_small)
        loss_centroid_medium = self.loss_centroid(pred_medium[:,0:1,...], centroid_medium)
        loss_centroid_large = self.loss_centroid(pred_large[:,0:1,...], centroid_large)
        loss_centroid = (loss_centroid_small + loss_centroid_medium + loss_centroid_large) / 3
        
        # get the encoding losses
        loss_encoding_small = self.loss_encoding(pred_small[:,1:,...], encoding_small, centroid_small)
        loss_encoding_medium = self.loss_encoding(pred_medium[:,1:,...], encoding_medium, centroid_medium)
        loss_encoding_large = self.loss_encoding(pred_large[:,1:,...], encoding_large, centroid_large)
        loss_encoding = (loss_encoding_small + loss_encoding_medium + loss_encoding_large) / 3
        
        loss = self.hparams.centroid_weight * loss_centroid + \
               self.hparams.encoding_weight * loss_encoding
               
        return {'test_loss': loss} 


    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
    
    def validation_step(self, batch, batch_idx):
        val_imgs = batch['image']
        
        centroid_small = batch['stride{0}/centroid_map'.format(self.hparams.strides[2])]
        encoding_small = batch['stride{0}/encoding_map'.format(self.hparams.strides[2])]
        
        centroid_medium = batch['stride{0}/centroid_map'.format(self.hparams.strides[1])]
        encoding_medium = batch['stride{0}/encoding_map'.format(self.hparams.strides[1])]
                                      
        centroid_large = batch['stride{0}/centroid_map'.format(self.hparams.strides[0])]
        encoding_large = batch['stride{0}/encoding_map'.format(self.hparams.strides[0])]
        
        pred_small, pred_medium, pred_large = self.forward(val_imgs)
        
        loss_centroid_small = self.loss_centroid(pred_small[:,0:1,...], centroid_small)
        loss_centroid_medium = self.loss_centroid(pred_medium[:,0:1,...], centroid_medium)
        loss_centroid_large = self.loss_centroid(pred_large[:,0:1,...], centroid_large)
        loss_centroid = (loss_centroid_small + loss_centroid_medium + loss_centroid_large) / 3
        
        # get the encoding losses
        loss_encoding_small = self.loss_encoding(pred_small[:,1:,...], encoding_small, centroid_small)
        loss_encoding_medium = self.loss_encoding(pred_medium[:,1:,...], encoding_medium, centroid_medium)
        loss_encoding_large = self.loss_encoding(pred_large[:,1:,...], encoding_large, centroid_large)
        loss_encoding = (loss_encoding_small + loss_encoding_medium + loss_encoding_large) / 3
        
        loss = self.hparams.centroid_weight * loss_centroid + \
               self.hparams.encoding_weight * loss_encoding
               
        return {'val_loss': loss} 

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = RAdam(self.network.parameters(), lr=self.hparams.learning_rate)
        return [opt], []

    def train_dataloader(self):
        if self.hparams.train_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.train_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_group=self.hparams.image_groups[0], strides=self.hparams.strides, norm_method=self.hparams.data_norm,\
                                        sh_order=self.hparams.sh_order)
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
    
    def test_dataloader(self):
        if self.hparams.test_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.test_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_group=self.hparams.image_groups[0], strides=self.hparams.strides, norm_method=self.hparams.data_norm, \
                                        sh_order=self.hparams.sh_order)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        if self.hparams.val_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.val_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_group=self.hparams.image_groups[0], strides=self.hparams.strides, norm_method=self.hparams.data_norm,\
                                        sh_order=self.hparams.sh_order)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)


    def on_epoch_end(self):
        
        # log sampled images
        z_slice_small = int(self.pred_small.shape[2]//2)        
        grid_small = torch.cat((self.pred_small[:,0:1,z_slice_small,:,:], self.centroid_small[:,0:1,z_slice_small,:,:]), 0)
        prediction_grid = torchvision.utils.make_grid(grid_small)
        self.logger.experiment.add_image('centroids_small', prediction_grid, self.current_epoch)
        
        z_slice_medium = int(self.pred_medium.shape[2]//2)   
        grid_small = torch.cat((self.pred_medium[:,0:1,z_slice_medium,:,:], self.centroid_medium[:,0:1,z_slice_medium,:,:]), 0)
        prediction_grid = torchvision.utils.make_grid(grid_small)
        self.logger.experiment.add_image('centroids_medium', prediction_grid, self.current_epoch)
        
        z_slice_large = int(self.pred_large.shape[2]//2) 
        grid_small = torch.cat((self.pred_large[:,0:1,z_slice_large,:,:], self.centroid_large[:,0:1,z_slice_large,:,:]), 0)
        prediction_grid = torchvision.utils.make_grid(grid_small)
        self.logger.experiment.add_image('centroids_large', prediction_grid, self.current_epoch)
        
        z_slice_raw = int(self.last_imgs.shape[2]//2)
        img_grid = torchvision.utils.make_grid(self.last_imgs[:,0,z_slice_raw,:,:])
        self.logger.experiment.add_image('raw_images', img_grid, self.current_epoch)
        
        
    def set_augmentations(self, augmentation_dict_file):
        self.augmentation_dict = json.load(open(augmentation_dict_file))
        
        
    @staticmethod
    def add_model_specific_args(parent_parser): 
        """
        Parameters you define here will be available to your model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--in_channels', default=1, type=int)
        parser.add_argument('--feat_channels', default=16, type=int)
        parser.add_argument('--patch_size', default=(64,128,128), type=int, nargs='+')
        parser.add_argument('--norm_method', default='instance', type=str)

        # data
        parser.add_argument('--data_norm', default='percentile', type=str)
        parser.add_argument('--data_root', default=r'D:\LfB\pytorchRepo\data\PNAS', type=str) 
        parser.add_argument('--train_list', default=r'D:\LfB\pytorchRepo\data\PNAS_harmonic_plant_split1_train.csv', type=str)
        parser.add_argument('--test_list', default=r'D:\LfB\pytorchRepo\data\PNAS_harmonic_plant_split1_test.csv', type=str)
        parser.add_argument('--val_list', default=r'D:\LfB\pytorchRepo\data\PNAS_harmonic_plant_split1_val.csv', type=str)
        parser.add_argument('--image_groups', default=('data/image',), type=str, nargs='+')
        parser.add_argument('--strides', default=(2,4,8), type=int, nargs='+')
        parser.add_argument('--sh_order', default=5, type=int)

        # training params (opt)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--centroid_weight', default=0.50, type=float)
        parser.add_argument('--encoding_weight', default=0.50, type=float)
        
        return parser