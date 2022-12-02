#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import napatrackmater.bTrackmate as TM
from pathlib import Path
from config_vollseg import VollSegConfig
import hydra
from hydra.core.config_store import ConfigStore


configstore = ConfigStore.instance()
configstore.store(name = 'VollSegConfig', node = VollSegConfig)

@hydra.main(config_path = 'conf', config_name = 'config_vollseg')
def main( config : VollSegConfig):
        #Trackmate writes an XML file of tracks, we use it as input
        xml_path = os.path.join(config.paths_vollseg.tracking_results_dir,config.files_vollseg.xml_filename) 
        #Path to Segmentation image for extracting any track information from labels 
        LabelImage = os.path.join(config.paths_vollseg.tracking_seg_image_dir, config.files_vollseg.tracking_seg_image)
        #Trackmate writes a spots and tracks file as csv
        spot_csv = os.path.join(config.paths_vollseg.tracking_results_dir,config.files_vollseg.spots_csv)
        track_csv = os.path.join(config.paths_vollseg.tracking_results_dir,config.files_vollseg.tracks_csv)
        savedir = config.paths_vollseg.tracking_results_dir
        Path(savedir).mkdir(exist_ok=True)
        scale = 255
        TM.import_TM_XML_Relabel(xml_path,LabelImage,spot_csv, track_csv, savedir, scale = scale)
        
if __name__=='__main__':
    
    main()        






