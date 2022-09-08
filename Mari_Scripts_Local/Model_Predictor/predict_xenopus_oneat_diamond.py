#!/usr/bin/env python
# coding: utf-8
import os
import glob
from oneat.NEATModels import NEATEynamic
from oneat.NEATUtils.utils import load_json
from pathlib import Path
from tifffile import imread

n_tiles = (2,2,2)
event_threshold = 0.9
event_confidence = 0.9
normalize = True
nms_function = 'iou'

imagedir = '/mnt/d/jean_zay_backup/Mari_Data_Oneat/raw/gt/'
segdir = '/mnt/d/jean_zay_backup/Mari_Data_Oneat/seg/'
model_dir = '/mnt/d/MariProject/Mari_Models/Oneat/'
savedir= '/mnt/d/jean_zay_backup/Mari_Data_Oneat/revolution_results/oneat_results/gt_d56_f64/'
model_name = 'Cellsplitdetectorxenopusvolumecnn_d56_f64'

remove_markers = False
division_categories_json = model_dir + 'Cellsplitdiamondcategoriesxenopus.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitdiamondcordxenopus.json'
cordconfig = load_json(division_cord_json)
model = NEATEynamic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)

for imagename in X:
     print(imagename)  
     print(imread(segdir + os.path.basename(os.path.splitext(imagename)[0]) + '.tif').shape)  
     marker_tree =  model.get_markers(imagename, 
                                                segdir
                                                )

                                   
     model.predict(imagename,
                           savedir, 
                           n_tiles = n_tiles, 
                           event_threshold = event_threshold, 
                           event_confidence = event_confidence,
                           marker_tree = marker_tree, 
                           remove_markers = remove_markers,
                        
                           nms_function = nms_function,
                           normalize = normalize)

