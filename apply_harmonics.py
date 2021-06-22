#!/usr/bin/env python3
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

import os
import numpy as np
import torch
from skimage import io
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label
from skimage.segmentation import watershed
from skimage.morphology import dilation, ball
from skimage.transform import resize
from argparse import ArgumentParser

from dataloader.harmonic_dataloader import MeristemH5Tiler as Tiler
from utils.harmonics import sampling2instance, harmonics2sampling, sampling2harmonics, instance2sampling, harmonic_non_max_suppression
from utils.utils import print_timestamp
from torch.autograd import Variable

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    
    """
    Main testing routine specific for this project
    :param hparams:
    """

    # ------------------------
    # 0 SANITY CHECKS
    # ------------------------
    if not isinstance(hparams.crop, (tuple, list)):
        hparams.crop = (hparams.crop,) * len(hparams.patch_size)
    assert all([p-2*c>0 for p,c in zip(hparams.patch_size, hparams.crop)]), 'Invalid combination of patch size and crop size.'

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = network(hparams=hparams)
    model = model.load_from_checkpoint(hparams.ckpt_path)
    model = model.cuda()
    
    # ------------------------
    # 2 INIT DATA TILER
    # ------------------------
    tiler = Tiler(hparams.test_list, hparams.data_root, patch_size=hparams.patch_size, overlap=hparams.overlap, crop=hparams.crop,\
                  image_group=hparams.image_groups, no_mask=True)
    os.makedirs(hparams.output_path, exist_ok=True)
    out_channels = int((hparams.sh_order+1)**2+1)
    fading_map = tiler.get_fading_map()
    
    # ------------------------
    # 3 INIT HARMONIC CONVERTER
    # ------------------------
    theta_phi_sampling = np.load(hparams.theta_phi_sampling)
    h2s = harmonics2sampling(hparams.sh_order, theta_phi_sampling)
    s2h = sampling2harmonics(hparams.sh_order, theta_phi_sampling)
    
    # ------------------------
    # 4 PROCESS EACH IMAGE
    # ------------------------
    for image_idx in range(len(tiler.data_list)):
        tiler.set_data_idx(image_idx)
        
        # Determine if the patch size exceeds the image size
        working_size = tuple(np.max(np.array(tiler.locations), axis=0) - np.min(np.array(tiler.locations), axis=0) + np.array(hparams.patch_size))
            
        # Initialize prediction map            
        pred_map = np.full((out_channels,)+working_size, 0, dtype=np.float32)    
        norm_map = np.full((1,)+working_size, 0, dtype=np.float32)      
                
        for patch_idx in range(tiler.__len__()):
            
            print_timestamp('Processing patch {0}/{1}...', [patch_idx+1, tiler.__len__()])
            
            # Get the mask
            sample = tiler.__getitem__(patch_idx)
            data = Variable(torch.from_numpy(sample['image'][np.newaxis,...]).cuda())
            data = data.float()
            
            # Predict the image
            pred_small, pred_medium, pred_large = model.forward(data)
            pred_small, pred_medium, pred_large = pred_small.cpu().data.numpy(), pred_medium.cpu().data.numpy(), pred_large.cpu().data.numpy()
            
            # Stack predictions (only batch size 1)
            pred_small = resize(pred_small[0,...], (out_channels,)+tiler.patch_size, order=1)
            pred_medium = resize(pred_medium[0,...], (out_channels,)+tiler.patch_size, order=1)
            pred_large = resize(pred_large[0,...], (out_channels,)+tiler.patch_size, order=1)
            pred = np.stack([pred_small,pred_medium,pred_large], axis=0)
                        
            # Convert to numpy array and compress all scales
            pred = np.average(pred, axis=0, weights=hparams.resolution_weights)   
            pred = pred.astype(np.float32)*fading_map
            
            # Add the predicted patch
            slicing = tuple(map(slice, (0,)+tuple(tiler.patch_start+tiler.global_crop_before), (out_channels,)+tuple(tiler.patch_end+tiler.global_crop_before)))
            pred_map[slicing] = pred_map[slicing] + pred
            norm_map[slicing] = norm_map[slicing] + fading_map
                    
        # Normalize the predicted centroids
        norm_map = np.clip(norm_map, 1e-5, np.inf)
        pred_map = pred_map / norm_map    
        
        # Crop the predicted image to its original size
        slicing_global = tuple(map(slice, (0,)+tuple(tiler.global_crop_before), (out_channels,)+tuple(tiler.global_crop_after)))
        pred_map = pred_map[slicing_global]
        
        # Filter predictions
        if hparams.centroid_thresh < 0: # optimize threshold
            
            threshs = np.arange(0.1,1,0.05)
            accuracies = np.zeros_like(threshs)
        
            # load centroid mask
            mask = tiler.get_whole_mask(mask_groups='data/stride2/centroid_map')[0,...]
            mask = resize(mask, tiler.data_shape, order=0)
            mask = label(mask)
            mask = dilation(mask, selem=ball(5))
            num_cells = np.max(mask)
            
            for num_thresh, thresh in enumerate(threshs):
                peak_map = peak_local_max(pred_map[0,...], threshold_abs=thresh, min_distance=hparams.minsize, exclude_border=False, indices=False)
                overlap = mask*peak_map
                accuracies[num_thresh] = len(np.unique(overlap)) / num_cells
                
                print_timestamp('Optimizing: {0:.3f} -> {1:.5f}', [thresh, accuracies[num_thresh]])
                
            # Take the last maximum threshold
            centroid_thresh = threshs[len(accuracies)-1 - np.argmax(np.flip(accuracies))]
            np.save(os.path.join(hparams.output_path, 'optimization_'+os.path.split(tiler.data_list[image_idx][0])[-1][:-3]), accuracies)
            
        else:
            centroid_thresh = hparams.centroid_thresh
        
        # Determine peaks
        peak_map = peak_local_max(pred_map[0,...], threshold_abs=centroid_thresh, min_distance=hparams.minsize, exclude_border=False, indices=False)
        peak_map = label(peak_map)
        


        ## WATERSHED
        
        if hparams.use_watershed:
            
            # Create Watershed segmentation
            ws_map = watershed(1-pred_map[0,...], markers=peak_map, mask=pred_map[0,...]>0.05)
            io.imsave(os.path.join(hparams.output_path, 'watershed_'+os.path.split(tiler.data_list[image_idx][0])[-1][:-3]+'.tif'), ws_map)
            
            # Get the centroid list
            regions = regionprops(ws_map)
            peaks = np.array([r.centroid for r in regions], dtype=np.int16)
            
        else:
            # Get peak coordinate list
            regions = regionprops(peak_map)
            peaks = np.array([r.centroid for r in regions], dtype=np.int16)
            
        
        ## HARMONICS
        
        # Convert encoded predictions to instances
        sh_sampling = np.zeros((len(peaks), h2s.num_coefficients), dtype=np.float16)
        sh_certainties = np.zeros((len(peaks),))
        for num_peak, peak in enumerate(peaks):
            print_timestamp('Converting instance {0}/{1}...',[num_peak+1, len(peaks)])
            
            # Get the predicted encodings and their certainties of the close neighbourhood
            nbhood_centroid = tuple(map(slice, (0,)+tuple(np.maximum(peak-2,0)),\
                                               (1,)+tuple(np.minimum(peak+3, tiler.data_shape))))
            certainties = pred_map[nbhood_centroid]
            certainties /= certainties.max()
            
            nbhood_encodings = tuple(map(slice, (1,)+tuple(np.maximum(peak-2,0)),
                                     (out_channels,)+tuple(np.minimum(peak+3, tiler.data_shape))))
            encodings = pred_map[nbhood_encodings]
            
            # Get average predicted encoding
            nbhood_size = np.prod(certainties.shape[1:])
            certainties = certainties.reshape((nbhood_size,))
            encodings = encodings.reshape((-1,nbhood_size))
            sh_sampling[num_peak,:] = np.average(encodings, axis=1, weights=certainties)
            sh_certainties[num_peak] = np.mean(certainties)
        
        # Convert harmonic encoding to spherical encoding 
        r_sampling = h2s.convert(sh_sampling)
        
        
        ## FILTERING
        
        if hparams.use_sizefilter:
            
            # Remove undersized cells
            peaks = peaks[sh_sampling[:,0]>=hparams.minsize,:]
            r_sampling = r_sampling[sh_sampling[:,0]>=hparams.minsize,:]
            sh_sampling = sh_sampling[sh_sampling[:,0]>=hparams.minsize,:]
            
            # Remove oversized cells
            peaks = peaks[sh_sampling[:,0]<=hparams.maxsize,:]
            r_sampling = r_sampling[sh_sampling[:,0]<=hparams.maxsize,:]
            sh_sampling = sh_sampling[sh_sampling[:,0]<=hparams.maxsize,:]
        
        if hparams.use_nms:
            peaks, r_sampling, sh_certainties = harmonic_non_max_suppression(peaks, sh_certainties, r_sampling, overlap_thresh=0.5)
        
        
        # reconstruct instance masks from spherical encodings
        peaks = np.array(peaks)
        r_sampling = np.array(r_sampling)
        instance_mask = sampling2instance(peaks, r_sampling, theta_phi_sampling, shape=tiler.data_shape)
                
        # Save the predicted instances
        instance_mask = instance_mask.astype(np.uint16)
        io.imsave(os.path.join(hparams.output_path, 'instances_'+os.path.split(tiler.data_list[image_idx][0])[-1][:-3]+'.tif'), instance_mask)
        io.imsave(os.path.join(hparams.output_path, 'harmonic0_'+os.path.split(tiler.data_list[image_idx][0])[-1][:-3]+'.tif'), pred_map[1,...])
        io.imsave(os.path.join(hparams.output_path, 'detections_'+os.path.split(tiler.data_list[image_idx][0])[-1][:-3]+'.tif'), pred_map[0,...])
        


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--output_path',
        type=str,
        default=r'HarmonicNet',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--ckpt_path',
        type=str,
        default=r'HarmonicNet/epoch=1999.ckpt',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--theta_phi_sampling',
        type=str,
        default=r'utils/theta_phi_sampling_5000points_10000iter.npy',
        help='path to the angular sampling file'
    )
    
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of GPUs to use'
    )
    
    
    parent_parser.add_argument(
        '--overlap',
        type=int,
        default=(10,10,10),
        help='overlap of adjacent patches',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--crop',
        type=int,
        default=(10,10,10),
        help='safety border crop of patches',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--resolution_weights',
        type=float,
        default=(0.01, 0.09, 0.9),
        help='weights for predictions at each resolution',
        nargs='+'
    )
    
    parent_parser.add_argument(
        '--centroid_thresh',
        type=float,
        default=-1,
        help='centroid threshold'
    )
        
    parent_parser.add_argument(
        '--minsize',
        type=int,
        default=5, # CElegans: 10 | Meristem: 20
        help='minimum size of cells'
    )
    
    parent_parser.add_argument(
        '--maxsize',
        type=int,
        default=30, # CElegans: 40 | Meristem: 75
        help='maximum size of cells'
    )
    
    parent_parser.add_argument(
        '--use_watershed',
        dest='use_watershed',
        action='store_true',
        help='uses watershed segmentation to remove possible false spherical shapes'
    )
    
    parent_parser.add_argument(
        '--use_sizefilter',
        dest='use_sizefilter',
        action='store_true',
        help='uses size estimations to remove possible false spherical shapes'
    )   
    
    parent_parser.add_argument(
        '--use_nms',
        dest='use_nms',
        action='store_true',
        help='uses non maximum suppression to remove dublicate spherical shapes'
    ) 
    
    parent_parser.add_argument(
        '--model',
        type=str,
        default='HarmonicNet',
        help='which model to load (HarmonicNet)'
    )
    
    
        
    parent_args = parent_parser.parse_known_args()[0]
    
    # load the desired network architecture
    if parent_args.model.lower() == 'harmonicnet':
        from models.HarmonicNet import HarmonicNet as network
    else:
        raise ValueError('Model {0} unknown.'.format(parent_args.model))
        
    # each LightningModule defines arguments relevant to it
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
    