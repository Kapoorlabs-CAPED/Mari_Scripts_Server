from config_oneat import OneatConfig
import hydra
from hydra.core.config_store import ConfigStore
from oneat.NEATUtils.utils import generate_membrane_locations
import numpy as np
import os

configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : OneatConfig):
    
    membranesegimage = os.path.join(config.paths_oneat.segdir, config.files_oneat.seg_image)
    csvfile = os.path.join(config.paths_oneat.savedir, config.files_oneat.pred_csv)
    savefile = os.path.join(config.paths_oneat.savedir, 'membrane_' + config.files_oneat.pred_csv)
    
    generate_membrane_locations(membranesegimage, csvfile, savefile)
     
     
     
