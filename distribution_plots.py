#!/usr/bin/env python
# coding: utf-8





import numpy as np
import os
import napatrackmater.bTrackmate as TM
from pathlib import Path
import hydra 
from config_vollseg import VollSegConfig
from hydra.core.config_store import ConfigStore 

configstore = ConfigStore.instance()
configstore.store(name = 'VollSegConfig', node = VollSegConfig)

@hydra.main(config_path = 'conf', config_name = 'config_vollseg')
def main():
        #Trackmate writes an XML file of tracks, we use it as input
        xml_path = os.path.join(config.paths_vollseg.tracking_results_dir,config.files_vollseg.xml_filename) 
        #Trackmate writes a spots and tracks file as csv
        spot_csv = os.path.join(config.paths_vollseg.tracking_results_dir,config.files_vollseg.spots_csv)
        track_csv = os.path.join(config.paths_vollseg.tracking_results_dir,config.files_vollseg.tracks_csv)
        savedir = config.paths_vollseg.tracking_results_dir
        Path(savedir).mkdir(exist_ok=True)
        TM.import_TM_XML_distplots(xml_path,spot_csv, track_csv, savedir)
        
if __name__=='__main__':
    
    main()        






