#!/usr/bin/env python

# coding: utf-8

from caped_ai_visualizations import visualize_activations
import hydra
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore
from oneat.NEATUtils.utils import load_json
from pathlib import Path 
import os
from tifffile import imread
configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : OneatConfig):
        imagedir = list(Path(config.paths_oneat.activation_image_dir).glob(config.params_predict.file_type))
        segimagedir = config.paths_oneat.activation_seg_dir
        model_dir = config.paths_oneat.model_dir
        categories_json = model_dir + config.trainclass.categories_json
        cord_json = model_dir + config.trainclass.cord_json
        
        catconfig = load_json(categories_json)
        cordconfig = load_json(cord_json)
        imagename = imagedir[0]
        image = imread(imagename)
        segimage = imread(os.path.join(segimagedir, Path(imagename).name))
        visualize_point = 5
        activations = visualize_activations(catconfig,cordconfig,
                              model_dir,
                              image,
                              segimage = segimage,
                              visualize_point = visualize_point,
                              oneat_vollnet = True)
        activations.VizualizePredictions()


if __name__=='__main__':
     main() 



