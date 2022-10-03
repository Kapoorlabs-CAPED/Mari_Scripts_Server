#!/usr/bin/env python
# coding: utf-8



from pathlib import Path
from oneat_augmentations import AugmentTZYXCsv
import os
from tifffile import imread, imwrite
import numpy as np
import hydra
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : OneatConfig):
        #Specify the directory containing images
        image_dir = Path(config.paths_oneat.train_basic_image_dir)
        #Specify the directory contaiing csv files
        csv_dir = Path(config.paths_oneat.train_basic_csv_dir)
        #Specify the directory containing the segmentations
        seg_image_dir = Path(config.paths_oneat.train_basic_seg_image_dir)
        #Specify the directory for storing the augmented images
        aug_image_dir = config.paths_oneat.train_image_dir
        #Specify the directory for storing the augmented labels
        aug_seg_image_dir = config.paths_oneat.train_seg_image_dir
        #Specify the directory for storing the augmented csv files
        aug_csv_dir = config.paths_oneat.train_csv_dir

        Path(aug_image_dir).mkdir(exist_ok = True)
        Path(aug_seg_image_dir).mkdir(exist_ok = True)
        Path(aug_csv_dir).mkdir(exist_ok = True)

        pattern = config.params_predict.file_type

        files_raw = list(image_dir.glob(pattern))
        files_seg = list(seg_image_dir.glob(pattern))
        files_csv = list(csv_dir.glob('*.csv'))

        #Name of the  events
        event_type_name = ["Normal", "Division"]
        #Label corresponding to event
        event_type_label = [0, 1]
        csv_name_diff = 'ONEAT'
        count = 0
        rotation_angles = [3,6,9]

        mean = 0.0
        sigma = 10.0
        distribution = 'Both'



        for fname in files_raw:
                        
            name = os.path.basename(os.path.splitext(fname)[0])   
            for Segfname in files_seg:

                Segname = os.path.basename(os.path.splitext(Segfname)[0])

                if name == Segname:
                    
                    
                    image = imread(fname).astype('uint8')
                    segimage = imread(Segfname).astype('uint16')

                    for csvfname in files_csv:
                            count = 0  
                            Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
                            for i in  range(0, len(event_type_name)):
                                event_name = event_type_name[i]
                                trainlabel = event_type_label[i]
                                classfound = (Csvname == csv_name_diff +  event_name + name)   
                                if classfound:
                                    
                                    #Rotate
                                    for rotate_angle in rotation_angles:
                                            
                                            rotate_pixels = AugmentTZYXCsv(rotate_angle = rotate_angle)

                                            aug_rotate_pixels,aug_rotate_pixels_label, aug_rotate_pixels_csv  = rotate_pixels.build(image = np.copy(image), labelimage = segimage, labelcsv = csvfname)
                                            
                                        
                                            save_name_raw = aug_image_dir + '/' + 'rotation_' +  str(rotate_angle) + name + '.tif'
                                            save_name_seg = aug_seg_image_dir + '/' + 'rotation_' +  str(rotate_angle) + name + '.tif'
                                            if os.path.exists(save_name_raw) == False:
                                                imwrite(save_name_raw, aug_rotate_pixels.astype('uint8'))
                                            if os.path.exists(save_name_seg) == False:    
                                                imwrite(save_name_seg, aug_rotate_pixels_label.astype('uint16'))
                                            aug_rotate_pixels_csv.to_csv(aug_csv_dir + '/' + csv_name_diff + event_name + 'rotation_' +  str(rotate_angle) + name +  '.csv', index = False, mode = 'w')
                                            count = count + 1
                    for csvfname in files_csv:
                            count = 0  
                            Csvname =  os.path.basename(os.path.splitext(csvfname)[0])
                            for i in  range(0, len(event_type_name)):
                                event_name = event_type_name[i]
                                trainlabel = event_type_label[i]
                                classfound = (Csvname == csv_name_diff +  event_name + name)   
                                if classfound: 
                                    #Additive Noise
                                    addnoise_pixels = AugmentTZYXCsv(mean = mean, sigma = sigma, distribution = distribution)

                                    aug_addnoise_pixels,aug_addnoise_pixels_label, aug_addnoise_pixels_csv  = addnoise_pixels.build(image = np.copy(image), labelimage = segimage, labelcsv = csvfname)
                                    
                                    save_name_raw = aug_image_dir + '/' + 'noise_' +  str(sigma) + name + '.tif'
                                    save_name_seg = aug_seg_image_dir + '/' + 'noise_' +   str(sigma) + name + '.tif'
                                    if os.path.exists(save_name_raw) == False:
                                        imwrite(save_name_raw, aug_addnoise_pixels.astype('uint8'))
                                    if os.path.exists(save_name_seg) == False:    
                                        imwrite(save_name_seg, aug_addnoise_pixels_label.astype('uint16'))
                                    aug_addnoise_pixels_csv.to_csv(aug_csv_dir + '/' + csv_name_diff + event_name + 'noise_' +   str(sigma) + name +  '.csv', index = False, mode = 'w')
                                    count = count + 1

                
if __name__=='__main__':
    main()