#!/usr/bin/env python
# coding: utf-8
import os
import glob
from oneat.NEATModels import NEATDynamic
from oneat.NEATUtils.utils import load_json
from pathlib import Path
import hydra
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : OneatConfig):
          n_tiles = config.params_predict.n_tiles
          event_threshold = config.params_predict.event_threshold
          event_confidence = config.params_predict.event_confidence
          downsamplefactor = config.params_predict.downsamplefactor
          start_project_mid = config.params_predict.start_project_mid
          end_project_mid = config.params_predict.end_project_mid
          normalize = config.params_predict.normalize
          nms_function = config.params_predict.nms_function

          imagedir = config.paths_oneat.imagedir
          segdir = config.paths_oneat.segdir
          model_dir = config.paths_oneat.model_dir
          savedir= config.paths_oneat.savedir
          model_name = config.files_oneat.model_name

          remove_markers = False
          division_categories_json = model_dir + config.files_oneat.categories_json
          catconfig = load_json(division_categories_json)
          division_cord_json = model_dir + config.files_oneat.lstm_cord_json
          cordconfig = load_json(division_cord_json)
          model = NEATDynamic(None, model_dir , model_name,catconfig, cordconfig)
          Path(savedir).mkdir(exist_ok=True)
          Raw_path = os.path.join(imagedir, config.params_predict.file_type)
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
               
if __name__=='__main__':
     main()               

