#!/usr/bin/env python
# coding: utf-8

import os

import napatrackmater.bTrackmate as TM
from pathlib import Path

from config_vollseg import VollSegConfig
import hydra 
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name= 'VollSegConfig', node= VollSegConfig)

@hydra.main(config_path='conf', config_name='config_vollseg')
def main():
        #Trackmate writes an XML file of tracks, we use it as input
        xml_path = os.path.join(config.paths_vollseg.tracking_results_dir,config.files_vollseg.xml_filename) 
        #Trackmate writes a spots and tracks file as csv
        spot_csv = os.path.join(config.paths_vollseg.tracking_results_dir,config.files_vollseg.spots_csv)
        links_csv = os.path.join(config.paths_vollseg.tracking_results_dir,config.files_vollseg.edges_csv)
        savedir = config.paths_vollseg.tracking_results_dir
        Path(savedir).mkdir(exist_ok=True)
        TM.import_TM_XML_statplots(xml_path,spot_csv, links_csv, savedir)
        
if __name__=='main':
    
    main()        






