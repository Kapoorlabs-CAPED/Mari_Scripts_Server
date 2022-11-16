#!/usr/bin/env python
# coding: utf-8

import os

from oneat.NEATModels import NEATVollNet, NEATTResNet, NEATLRNet, NEATDenseVollNet
from oneat.NEATModels.config import volume_config, lstm_config
from oneat.NEATUtils.utils import save_json, load_json
import hydra
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore
os.environ["CUDA_VISIBLE_DEVICES"]="1"
configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : OneatConfig):
    npz_directory = config.paths_oneat.npz_directory
    model_dir = config.paths_oneat.model_dir
    npz_name = config.files_oneat.npz_name
    npz_val_name = config.files_oneat.npz_val_name
    #Neural network parameters
    division_categories_json = os.path.join(model_dir, config.trainclass.categories_json)
    key_categories = load_json(division_categories_json)
    
    division_cord_json = os.path.join(model_dir, config.trainclass.cord_json)
    key_cord = load_json(division_cord_json)

    #Number of starting convolutional filters, is doubled down with increasing depth
    startfilter = config.params_train.startfilter
    #CNN network start layer, mid layers and lstm layer kernel size
    start_kernel = config.params_train.start_kernel
    mid_kernel = config.params_train.mid_kernel
    #Size of the gradient descent length vector, start small and use callbacks to get smaller when reaching the minima
    learning_rate = config.params_train.learning_rate
    #For stochastic gradient decent, the batch size used for computing the gradients
    batch_size = config.params_train.batch_size
    #Training epochs, longer the better with proper chosen learning rate
    epochs = config.params_train.epochs
    
    #The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
    show = config.params_train.show
    stage_number = config.params_train.stage_number
    size_tminus = config.params_train.size_tminus
    size_tplus = config.params_train.size_tplus
    imagex = config.params_train.imagex
    imagey = config.params_train.imagey
    imagez = config.params_train.imagez
    trainclass = eval(config.trainclass.training_class)
    trainconfig = eval(config.trainclass.training_config)
    depth = dict(config.params_train.depth)
    reduction = config.params_train.reduction
    config = trainconfig(npz_directory = npz_directory, npz_name = npz_name, npz_val_name = npz_val_name,  
                            key_categories = key_categories, key_cord = key_cord, imagex = imagex,
                            reduction = reduction,
                            imagey = imagey, imagez = imagez, size_tminus = size_tminus, size_tplus = size_tplus, epochs = epochs,learning_rate = learning_rate,
                            depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel, stage_number = stage_number,
                            show = show,startfilter = startfilter, batch_size = batch_size)

    config_json = config.to_json()
    print(config)
    save_json(config_json, model_dir + '/' + 'parameters.json')

    Train = trainclass(config, model_dir, key_categories, key_cord)
    Train.loadData()
    Train.TrainModel()

if __name__ == '__main__':
    main()  




