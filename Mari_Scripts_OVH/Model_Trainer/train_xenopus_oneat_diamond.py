#!/usr/bin/env python
# coding: utf-8

import os
from config_oneat import TrainOneatConfig
from oneat.NEATModels import NEATEynamic
from oneat.NEATModels.config import diamond_config
from oneat.NEATUtils.utils import save_json, load_json
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'TrainOneatConfig', node = TrainOneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : TrainOneatConfig):
    npz_directory = config.paths.npz_directory
    model_dir = config.paths.model_dir
    model_name = config.files.model_name
    npz_name = config.files.npz_name
    npz_val_name = config.files.npz_val_name
    #Neural network parameters
    division_categories_json = model_dir + config.files.vollnet_categories_json
    key_categories = load_json(division_categories_json)
    division_cord_json = model_dir + config.files.vollnet_cord_json
    key_cord = load_json(division_cord_json)


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


    config = diamond_config(npz_directory =npz_directory, npz_name = npz_name, npz_val_name = npz_val_name, 
                            key_categories = key_categories, key_cord = key_cord, nboxes = nboxes, imagex = imagex,
                            imagey = imagey, imagez = imagez, size_tminus = size_tminus, size_tplus = size_tplus, epochs = epochs,learning_rate = learning_rate,
                            depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel, stage_number = stage_number,
                            show = show,startfilter = startfilter, batch_size = batch_size, model_name = model_name)

    config_json = config.to_json()
    print(config)
    save_json(config_json, model_dir + os.path.splitext(model_name)[0] + '_Parameter.json')

    Train = NEATEynamic(config, model_dir, model_name)
    Train.loadData()
    Train.TrainModel()






