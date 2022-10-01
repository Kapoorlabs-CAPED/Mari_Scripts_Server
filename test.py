#!/usr/bin/env python
# coding: utf-8

import os

from oneat.NEATModels import NEATCynamic, NEATDynamic, NEATEynamic, NEATLDynamic, NEATSDynamic, NEATFocus, NEATSynamic
from oneat.NEATModels.config import diamond_config, dynamic_config
from oneat.NEATUtils.utils import save_json, load_json
import hydra
from config_oneat import TrainOneatConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'TrainOneatConfig', node = TrainOneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : TrainOneatConfig):
    
    
    npz_directory = config.paths_oneat.npz_directory
    model_dir = config.paths_oneat.model_dir
    model_name = config.files_oneat.model_name
    npz_name = config.files_oneat.npz_name
    npz_val_name = config.files_oneat.npz_val_name
   


    #Number of starting convolutional filters, is doubled down with increasing depth
    startfilter = config.params.startfilter
    #CNN network start layer, mid layers and lstm layer kernel size
    start_kernel = config.params.start_kernel
    mid_kernel = config.params.mid_kernel
    #Network depth has to be 9n + 2, n= 3 or 4 is optimal for Notum dataset
    depth = config.params.depth
    #Size of the gradient descent length vector, start small and use callbacks to get smaller when reaching the minima
    learning_rate = config.params.learning_rate
    #For stochastic gradient decent, the batch size used for computing the gradients
    batch_size = config.params.batch_size
    #Training epochs, longer the better with proper chosen learning rate
    epochs = config.params.epochs
    nboxes = config.params.nboxes
    #The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
    show = config.params.show
    stage_number = config.params.stage_number
    size_tminus = config.params.size_tminus
    size_tplus = config.params.size_tplus
    imagex = config.params.imagex
    imagey = config.params.imagey
    imagez = config.params.imagez
    trainclass = eval(config.trainclass.training_class)
    trainconfig = eval(config.trainclass.training_config)

    config = trainconfig(npz_directory =npz_directory, npz_name = npz_name, npz_val_name = npz_val_name, 
                            key_categories = {0:0}, key_cord = {0:0}, nboxes = nboxes, imagex = imagex,
                            imagey = imagey, imagez = imagez, size_tminus = size_tminus, size_tplus = size_tplus, epochs = epochs,learning_rate = learning_rate,
                            depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel, stage_number = stage_number,
                            show = show,startfilter = startfilter, batch_size = batch_size, model_name = model_name)

    print(config)
    Train = trainclass(config, model_dir, model_name)
    print(Train)
 

if __name__ == '__main__':
    main()  




