#!/usr/bin/env python
# coding: utf-8
import os
import glob
from oneat.NEATModels import NEATCynamic
from oneat.NEATUtils.utils import load_json
from pathlib import Path

n_tiles = (1,1,1)
event_threshold = 0.9
event_confidence = 0.9
downsamplefactor = 1
#For a Z of 0 to 22 this setup takes the slices from 11 - 4 = 7 to 11 + 1 = 12
start_project_mid = 4
end_project_mid = 1
normalize = True
nms_function = 'iou'

imagedir = '/mnt/WorkHorse/Mari_Data_Oneat/raw/gt/'
segdir = '/mnt/WorkHorse/Mari_Data_Oneat/seg/'
model_dir = '/mnt/WorkHorse/Mari_Models/Oneat/'
savedir= '/mnt/WorkHorse/Mari_Data_Oneat/revolution_results/oneat_results/gt_scalarcnn_d56_f64/'
model_name = 'Cellsplitdetectorxenopusscalarcnn_d56_f64'

remove_markers = False
division_categories_json = model_dir + 'Cellsplitcategoriesxenopus.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitcordxenopus.json'
cordconfig = load_json(division_cord_json)
model = NEATCynamic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)

for imagename in X:
     print(imagename)   
     marker_tree =  model.get_markers(imagename, 
                                                segdir,
                                                start_project_mid = start_project_mid,
                                                end_project_mid = end_project_mid,  
                                                )

                                   
     model.predict(imagename,
                           savedir, 
                           n_tiles = n_tiles, 
                           event_threshold = event_threshold, 
                           event_confidence = event_confidence,
                           marker_tree = marker_tree, 
                           remove_markers = remove_markers,
                           nms_function = nms_function,
                           downsamplefactor = downsamplefactor,
                           start_project_mid = start_project_mid,
                           end_project_mid = end_project_mid,
                           normalize = normalize)

