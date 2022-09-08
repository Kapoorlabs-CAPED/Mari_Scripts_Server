#!/usr/bin/env python
# coding: utf-8

import os
from oneat.NEATModels import NEATCynamic
from oneat.NEATModels.config import dynamic_config
from oneat.NEATUtils.helpers import save_json, load_json


npz_directory = '/mnt/d/jean_zay_backup/Mari_Data_Training/oneat_training/oneat_train_patches_m1p1/'
npz_name = 'Xenopus_oneat_training_m1p1_aug.npz'
npz_val_name = 'Xenopus_oneat_training_m1p1_augval.npz'

#Read and Write the h5 file, directory location and name
model_dir =  '/mnt/d/jean_zay_backup/Mari_Models/Oneat/'
model_name = 'Cellsplitdetectorxenopusscalarcnn_d56_f48.h5'



#Neural network parameters
division_categories_json = model_dir + 'Cellsplitcategoriesxenopus.json'
key_categories = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitcordxenopus.json'
key_cord = load_json(division_cord_json)

#For ORNET use residual = True and for OSNET use residual = False
residual = True
#Number of starting convolutional filters, is doubled down with increasing depth
startfilter = 48
#CNN network start layer, mid layers and lstm layer kernel size
start_kernel = 7
mid_kernel = 3
#Network depth has to be 9n + 2, n= 3 or 4 is optimal for Notum dataset
depth = 56
#Size of the gradient descent length vector, start small and use callbacks to get smaller when reaching the minima
learning_rate = 1.0E-3
#For stochastic gradient decent, the batch size used for computing the gradients
batch_size = 20
# use softmax for single event per box, sigmoid for multi event per box
#Training epochs, longer the better with proper chosen learning rate
epochs = 250
nboxes = 1
#The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
show = False
stage_number = 3
size_tminus = 1
size_tplus = 1
imagex = 64
imagey = 64






config = dynamic_config(npz_directory =npz_directory, npz_name = npz_name, npz_val_name = npz_val_name, 
                         key_categories = key_categories, key_cord = key_cord, nboxes = nboxes, imagex = imagex,
                         imagey = imagey, size_tminus = size_tminus, size_tplus = size_tplus, epochs = epochs, learning_rate = learning_rate,
                         residual = residual, depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel, stage_number = stage_number, show = show, startfilter = startfilter, batch_size = batch_size, model_name = model_name)

config_json = config.to_json()

print(config)
save_json(config_json, model_dir + os.path.splitext(model_name)[0] + '_Parameter.json')




Train = NEATCynamic(config, model_dir, model_name)

Train.loadData()

Train.TrainModel()





