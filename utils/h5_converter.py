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
import h5py
import glob
import numpy as np

from skimage import io, morphology, measure, filters
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial import ConvexHull, Delaunay

from utils.utils import print_timestamp
from utils.harmonics import instance2sampling, sampling2harmonics



def h5_writer(data_list, save_path, group_root='data', group_names=['data']):
    
    save_path = os.path.abspath(save_path)
    
    assert(len(data_list)==len(group_names)), 'Each data matrix needs a group name'
    
    with h5py.File(save_path, 'w') as f_handle:
        grp = f_handle.create_group(group_root)
        for data, group_name in zip(data_list, group_names):
            grp.create_dataset(group_name, data=data, chunks=True, compression='gzip')




def h52tif(file_dir='', group_names=['data/image']):
    
    # Get all files within the given directory
    filelist = glob.glob(os.path.join(file_dir, '*.h5'))
    
    # Create saving folders
    for group in group_names:
        os.makedirs(os.path.join(file_dir, ''.join(s for s in group if s.isalnum())), exist_ok=True)
    
    # Save each desired group
    for num_file,file in enumerate(filelist):
        print_timestamp('Processing file {0}/{1}', (num_file+1, len(filelist)))
        with h5py.File(file, 'r') as file_handle:
            for group in group_names:
                data = file_handle[group][:]
                io.imsave(os.path.join(file_dir, ''.join(s for s in group if s.isalnum()), os.path.split(file)[-1][:-2]+'tif'), data)



def replace_h5_group(source_list, target_list, source_group='data/image', target_group=None):
    
    assert len(target_list)==len(source_list), 'There needs to be one target ({0}) for each source ({1})!'.format(len(target_list), len(source_list))
    if target_group is None: target_group=source_group

    for num_pair, pair in enumerate(zip(source_list, target_list)):
        print_timestamp('Processing file {0}/{1}...', [num_pair+1, len(target_list)])
        
        # Load the source mask
        with h5py.File(pair[0], 'r') as source_handle:
            source_data = source_handle[source_group][...]
            
        # Save the data to the target file
        with h5py.File(pair[1], 'r+') as target_handle:
            target_data = target_handle[target_group]
            target_data[...] = source_data


            
def add_h5_group(source_list, target_list, source_group='data/distance', target_group=None):
    
    assert len(target_list)==len(source_list), 'There needs to be one target ({0}) for each source ({1})!'.format(len(target_list), len(source_list))
    if target_group is None: target_group=source_group
    
    for num_pair, pair in enumerate(zip(source_list, target_list)):
        
        print_timestamp('Processing file {0}/{1}...', [num_pair+1, len(source_list)])
            
        # Get the data from the source file
        with h5py.File(pair[0], 'r') as source_handle:
            source_data = source_handle[source_group][...]
            
        # Save the data to the target file
        with h5py.File(pair[1], 'a') as target_handle:
            target_handle.create_dataset(target_group, data=source_data, chunks=True, compression='gzip')



def remove_h5_group(file_list, remove_group='data/nuclei'):
    
    for num_file, file in enumerate(file_list):
        
        print_timestamp('Processing file {0}/{1}...', [num_file+1, len(file_list)])
        
        with h5py.File(file, 'a') as file_handle:
            del file_handle[remove_group]

            
            
def flood_fill_hull(image):    
    
    points = np.transpose(np.where(image))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    
    return out_img


        

def rescale_data(data, zoom_factor, order=0):
    
    if any([zf!=1 for zf in zoom_factor]):
        data_shape = data.shape
        data = zoom(data, zoom_factor, order=3)
        print_timestamp('Rescaled image from size {0} to {1}'.format(data_shape, data.shape))
        
    return data
        


def prepare_images(data_path='', folders=[''], identifier='*.tif', descriptor='', normalize=[0,100],\
                   get_distance=False, get_illumination=False, fg_selem_size=5, zoom_factor=(1,1,1), channel=0):

    data_path = os.path.abspath(data_path)
     
    for image_folder in folders:
        image_list = glob.glob(os.path.join(data_path, image_folder, identifier))
        for num_file,file in enumerate(image_list):
            
            print_timestamp('Processing image {0}/{1} in "{2}"',(num_file+1, len(image_list), image_folder))
                            
            # load the image
            processed_img = io.imread(file)     
            processed_img = processed_img.astype(np.float32)
            
            # get the desired channel, if the image is a multichannel image
            if processed_img.ndim == 4:
                processed_img = processed_img[...,channel]
            
            # rescale the image
            processed_img = rescale_data(processed_img, zoom_factor, order=3)
            
            # normalize the image
            perc1, perc2 = np.percentile(processed_img, list(normalize))
            processed_img -= perc1
            processed_img /= (perc2-perc1)
            processed_img = np.clip(processed_img, 0, 1)
            processed_img = processed_img.astype(np.float32)
                
            save_imgs = [processed_img,]
            save_groups = ['image',]
            
            if get_illumination:
                
                # create downscales image for computantially intensive processing
                small_img = processed_img[::4,::4,::4]
                
                # create an illuminance image (downscale for faster processing)
                illu_img = morphology.closing(small_img, selem=morphology.ball(7))
                illu_img = filters.gaussian(illu_img, 2).astype(np.float32)
                
                # rescale illuminance image
                illu_img = np.repeat(illu_img, 4, axis=0)
                illu_img = np.repeat(illu_img, 4, axis=1)
                illu_img = np.repeat(illu_img, 4, axis=2)
                dim_missmatch = np.array(processed_img.shape)-np.array(illu_img.shape)
                if dim_missmatch[0]<0: illu_img = illu_img[:dim_missmatch[0],...]
                if dim_missmatch[1]<0: illu_img = illu_img[:,:dim_missmatch[1],:]
                if dim_missmatch[2]<0: illu_img = illu_img[...,:dim_missmatch[2]]
                
                save_imgs.append(illu_img.astype(np.float32))
                save_groups.append('illumination')
                
            if get_distance:
                
                # create downscales image for computantially intensive processing
                small_img = processed_img[::4,::4,::4]
                
                # find suitable threshold
                thresh = filters.threshold_otsu(small_img)
                fg_img = small_img > thresh
                
                # remove noise and fill holes
                fg_img = morphology.binary_closing(fg_img, selem=morphology.ball(fg_selem_size))
                fg_img = morphology.binary_opening(fg_img, selem=morphology.ball(fg_selem_size))
                fg_img = flood_fill_hull(fg_img)
                fg_img = fg_img.astype(np.bool)
                
                # create distance transform
                fg_img = distance_transform_edt(fg_img) - distance_transform_edt(~fg_img)
                
                # rescale distance image
                fg_img = np.repeat(fg_img, 4, axis=0)
                fg_img = np.repeat(fg_img, 4, axis=1)
                fg_img = np.repeat(fg_img, 4, axis=2)
                dim_missmatch = np.array(processed_img.shape)-np.array(fg_img.shape)
                if dim_missmatch[0]<0: fg_img = fg_img[:dim_missmatch[0],...]
                if dim_missmatch[1]<0: fg_img = fg_img[:,:dim_missmatch[1],:]
                if dim_missmatch[2]<0: fg_img = fg_img[...,:dim_missmatch[2]]
                
                save_imgs.append(fg_img.astype(np.float32))
                save_groups.append('distance')
            
            # save the data
            save_name = os.path.split(file)[-1]
            save_name = os.path.join(data_path, image_folder, descriptor+save_name[:-4]+'.h5')
            h5_writer(save_imgs, save_name, group_root='data', group_names=save_groups)
      
      
     
    


def prepare_harmonics(data_path='data_root', sampling_file=r'utils/theta_phi_sampling_5000points_10000iter.npy', bg_label=0,\
                      folders=['experiment1', 'experiment2'], identifier='*.tif', descriptor='harmonic_encoding', zoom_factor=(1,1,1), strides=[2,4,8]):
     
    data_path = os.path.abspath(data_path)
    
    # get additional harmonic encoding
    if prepare_harmonics:
            
        for mask_folder in folders:
            theta_phi_sampling = np.load(sampling_file)
            s2h_converter = sampling2harmonics(11, theta_phi_sampling)
            
            mask_list = glob.glob(os.path.join(data_path, mask_folder, identifier))
            os.makedirs(os.path.join(data_path, mask_folder, 'harmonic_encoding'), exist_ok=True)
            for num_file,file in enumerate(mask_list):
                
                print('Converting mask {0}/{1} in {2} to harmonic encodings'.format(num_file+1, len(mask_list), mask_folder))
                
                instance_mask = io.imread(file)
                instance_mask = instance_mask.astype(np.uint16)
                
                # rescale the mask
                instance_mask = rescale_data(instance_mask, zoom_factor, order=0)
                    
                labels, centroids, sampling = instance2sampling(instance_mask, theta_phi_sampling, bg_label=bg_label, verbose=True)
                harmonic_encodings = s2h_converter.convert(sampling)
                
                centroid_maps = []
                encoding_maps = []
                
                for stride in strides:
                    
                    scale_shape = tuple(np.ceil(np.array(instance_mask.shape)/stride).astype(np.int))
                    
                    instances_strided = instance_mask[::stride, ::stride, ::stride]
                    
                    centroid_map = np.zeros(scale_shape, dtype=np.float32)
                    encoding_map = np.zeros(scale_shape+(harmonic_encodings.shape[1],), dtype=np.float16)                    
                    
                    for label, encoding in zip(labels, harmonic_encodings):
                        if label == bg_label: continue
                    
                        # create a normalized distance map of the current cell
                        cell_map = np.zeros_like(centroid_map)
                        cell_map[instances_strided==label] = 1
                        cell_map = distance_transform_edt(cell_map)
                        if np.abs(cell_map.max()) > 0:
                            cell_map /= cell_map.max()
                            
                        # add the distance map to the overall map
                        centroid_map = centroid_map + cell_map
                        
                        # at each position of the cell, insert the encoding
                        x_cell,y_cell,z_cell = np.where(instances_strided==label)
                        for x,y,z in zip(x_cell,y_cell,z_cell):
                            encoding_map[tuple([x,y,z])] = encoding
                        
                    centroid_map = centroid_map.astype(np.float32)
                        
                    # extend the current map lists
                    centroid_maps.append(centroid_map)
                    encoding_maps.append(encoding_map)
                        
                # save the data
                save_name = os.path.split(file)[-1]
                save_name = os.path.join(data_path, mask_folder, 'harmonic_encoding', descriptor+save_name[:-4]+'.h5')
                h5_writer(centroid_maps+encoding_maps, save_name, group_root='data', group_names=\
                          ['stride{0}/centroid_map'.format(s) for s in strides]+\
                          ['stride{0}/encoding_map'.format(s) for s in strides])
                    
    
        