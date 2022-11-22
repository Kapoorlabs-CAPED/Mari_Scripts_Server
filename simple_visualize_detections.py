#!/usr/bin/env python

# coding: utf-8

from oneat.NEATUtils import VizDet
import hydra
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore
import os
configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : OneatConfig):
        
        event_threshold = config.params_predict.event_threshold
        VizDet()

if __name__=='__main__':
     main() 



