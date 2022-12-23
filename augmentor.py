import os
import glob
from tifffile import imread, imwrite
from caped_ai_augmentations import AugmentYX
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from config_vollseg import VollSegConfig
import hydra
from hydra.core.config_store import ConfigStore


configstore = ConfigStore.instance()
configstore.store(name = 'VollSegConfig', node = VollSegConfig)

@hydra.main(config_path = 'conf', config_name = 'config_vollseg')
def main( config : VollSegConfig):
        image_dir =  Path(os.path.join(config.paths_vollseg.base_dir, config.paths_vollseg.raw_dir))
        label_dir = Path(os.path.join(config.paths_vollseg.base_dir, config.paths_vollseg.binary_mask_dir))

        aug_image_dir =  os.path.join(config.paths_vollseg.base_dir, config.paths_vollseg.aug_raw_dir)
        aug_seg_image_dir = os.path.join(config.paths_vollseg.base_dir, config.paths_vollseg.aug_binary_mask_dir)
        acceptable_formats = config.aug_params.pattern

        Path(aug_image_dir).mkdir(exist_ok=True)
        Path(aug_seg_image_dir).mkdir(exist_ok=True)

        gauss_filter_size = config.aug_params.gauss_filter_size
        #choices for augmentation below are 1 or 2 or None
        flip_axis= config.aug_params.flip_axis
        shift_axis= config.aug_params.shift_axis
        zoom_axis= config.aug_params.zoom_axis
        #shift range can be between -1 and 1 (-1 and 1 will translate the pixels completely out), zoom range > 0
        shift_range= config.aug_params.shift_range
        zoom_range= config.aug_params.zoom_range
        rotate_axis= config.aug_params.rotate_axis
    
        rotation_angles = config.aug_params.rotation_angles
        sigma = config.aug_params.sigma
        mean = config.aug_params.mean
        alpha_affine = config.aug_params.alpha_affine
        alpha = config.aug_params.alpha
        distribution = config.aug_params.distribution
        
        count = 0
        for fname in os.listdir(image_dir):

            for secondfname in os.listdir(label_dir):
                if any(fname.endswith(f) for f in acceptable_formats):
                    name = os.path.basename(os.path.splitext(fname)[0])
                    LabelName = os.path.basename(os.path.splitext(secondfname)[0])
                    if name == LabelName:
                        image = imread(fname)
                    
                        labelimage = imread(secondfname)
                        for rotate_angle in rotation_angles:
                                        
                                        rotate_pixels = Augmentation2D(rotate_angle = rotate_angle)

                                        aug_rotate_pixels,aug_rotate_pixels_label  = rotate_pixels.build(image = np.copy(image), labelimage = labelimage)
                                        
                                    
                                        save_name_raw = aug_image_dir + '/' + 'rotation_' +  str(rotate_angle) + name + '.tif'
                                        save_name_seg = aug_seg_image_dir + '/' + 'rotation_' +  str(rotate_angle) + name + '.tif'
                                        if os.path.exists(save_name_raw) == False:
                                            imwrite(save_name_raw, aug_rotate_pixels.astype('float32'))
                                        if os.path.exists(save_name_seg) == False:    
                                            imwrite(save_name_seg, aug_rotate_pixels_label.astype('uint16'))
                                        count = count + 1   

                        addnoise_pixels = Augmentation2D(mean = mean, sigma = sigma, distribution = distribution)

                        aug_addnoise_pixels,aug_addnoise_pixels_label  = addnoise_pixels.build(image = np.copy(image), labelimage = labelimage)
                        
                        save_name_raw = aug_image_dir + '/' + 'noise_' +  str(sigma) + name + '.tif'
                        save_name_seg = aug_seg_image_dir + '/' + 'noise_' +   str(sigma) + name + '.tif'
                        if os.path.exists(save_name_raw) == False:
                            imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                        if os.path.exists(save_name_seg) == False:    
                            imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                        count = count + 1                
        
                        adddeform_pixels = Augmentation2D(alpha_affine = alpha_affine, alpha = alpha, sigma = sigma)

                        aug_adddeform_pixels,aug_adddeform_pixels_label  = adddeform_pixels.build(image = np.copy(image), labelimage = labelimage)
                        
                        save_name_raw = aug_image_dir + '/' + 'deform_' +  str(sigma) + name + '.tif'
                        save_name_seg = aug_seg_image_dir + '/' + 'deform_' +   str(sigma) + name + '.tif'
                        if os.path.exists(save_name_raw) == False:
                            imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                        if os.path.exists(save_name_seg) == False:    
                            imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                        count = count + 1  

                        flip_pixels = Augmentation2D(vertical_flip = True)

                        aug_flip_pixels,aug_flip_pixels_label  = flip_pixels.build(image = np.copy(image), labelimage = labelimage)
                        
                        save_name_raw = aug_image_dir + '/' + 'vflip_'  + name + '.tif'
                        save_name_seg = aug_seg_image_dir + '/' + 'vflip_'  + name + '.tif'
                        if os.path.exists(save_name_raw) == False:
                            imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                        if os.path.exists(save_name_seg) == False:    
                            imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                        count = count + 1

                        flip_pixels = Augmentation2D(horizontal_flip = True)

                        aug_flip_pixels,aug_flip_pixels_label  = flip_pixels.build(image = np.copy(image), labelimage = labelimage)
                        
                        save_name_raw = aug_image_dir + '/' + 'hflip_'  + name + '.tif'
                        save_name_seg = aug_seg_image_dir + '/' + 'hflip_'  + name + '.tif'
                        if os.path.exists(save_name_raw) == False:
                            imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                        if os.path.exists(save_name_seg) == False:    
                            imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                        count = count + 1
                        
                        
if __name__=='__main__':
    
    main()                           