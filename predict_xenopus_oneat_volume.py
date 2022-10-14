#!/usr/bin/env python
# coding: utf-8
import os
import glob
from oneat.NEATModels import NEATVollNet, NEATTResNet, NEATLRNet
from oneat.NEATUtils.utils import load_json
from pathlib import Path
from tifffile import imread
import hydra
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore
import numpy as np
configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : OneatConfig):
          n_tiles = config.params_predict.n_tiles
          event_threshold = config.params_predict.event_threshold
          event_confidence = config.params_predict.event_confidence
          normalize = config.params_predict.normalize
          nms_function = config.params_predict.nms_function
          
          imagedir = config.paths_oneat.imagedir
          segdir = config.paths_oneat.segdir
          model_dir = config.paths_oneat.model_dir
          savedir= config.paths_oneat.savedir
          

          remove_markers = config.params_predict.remove_markers
          division_categories_json = model_dir + config.trainclass.categories_json
          catconfig = load_json(division_categories_json)
          division_cord_json = model_dir + config.trainclass.cord_json
          cordconfig = load_json(division_cord_json)
          training_class = eval(config.trainclass.training_class)
          model = training_class(None, model_dir, catconfig, cordconfig)
          Path(savedir).mkdir(exist_ok=True)
          Raw_path = os.path.join(imagedir, config.params_predict.file_type)
          X = glob.glob(Raw_path)

          for imagename in X:
               image = imread(imagename).astype(np.uint8)
               segimage = os.path.join(segdir, Path(imagename).name)  
               marker_tree =  model.get_markers(segimage)
                                             
               model.predict(image,
                             savedir, 
                             n_tiles = n_tiles, 
                             event_threshold = event_threshold, 
                             event_confidence = event_confidence,
                             marker_tree = marker_tree, 
                             remove_markers = remove_markers,
                             nms_function = nms_function,
                             normalize = normalize)
               
               
if __name__=="__main__":
     main()
