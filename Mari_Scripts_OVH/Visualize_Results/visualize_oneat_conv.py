#!/usr/bin/env python
# coding: utf-8
import os
import glob
from vnet import VizOneat
from oneat.NEATUtils.utils import load_json
from pathlib import Path
from tifffile import imread

n_tiles = (1,1,1)
event_threshold = 0.9
event_confidence = 0.9
normalize = True
nms_function = 'iou'

imagedir = '/mnt/WorkHorse/Mari_Data_Oneat/raw/gt/viz/'
model_dir = '/mnt/WorkHorse/Mari_Models/StarDist3D/'
model_name = 'Carcinoma'

#division_categories_json = model_dir + 'Cellsplitdiamondcategoriesxenopus.json'
#catconfig = load_json(division_categories_json)
#division_cord_json = model_dir + 'Cellsplitdiamondcordxenopus.json'
#cordconfig = load_json(division_cord_json)

Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)

for imagename in X:
     print(imagename)  
     model = VizOneat(None, imagename, model_dir , model_name, catconfig = None, cordconfig = None,oneat_vollnet = False, voll_care = True)
     model.PrepareNet()
     model.VizActivations()

