#!/usr/bin/env python

# coding: utf-8

from oneat.NEATUtils import NEATViz
import hydra
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : OneatConfig):
        imagedir = config.paths_oneat.imagedir
        segimagedir = config.paths_oneat.segdir
        csvdir = config.paths_oneat.savedir
        model_dir = config.paths_oneat.model_dir
        savedir = config.paths_oneat.savedir + 'Clean_CSV/' 
        categories_json = model_dir + config.trainclass.categories_json
        fileextension = config.params_predict.file_type
        event_threshold = config.params_predict.event_threshold
        nms_space = config.params_predict.nms_space
        nms_time = config.params_predict.nms_time
        volume = config.params_predict.volume 
        NEATViz(imagedir,
                csvdir,
                savedir,
                categories_json,
                segimagedir = segimagedir,
                fileextension = fileextension,
                volume = volume ,
                event_threshold = event_threshold,
                nms_space = nms_space,
                nms_time = nms_time)


if __name__=='__main__':
     main() 



