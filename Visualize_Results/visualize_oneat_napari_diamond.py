#!/usr/bin/env python

# coding: utf-8


from oneat.NEATUtils import NEATViz




imagedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/raw/gt/'
segimagedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/seg/'
csvdir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/revolution_results/oneat_results/gt_diamond_92/'
savedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/revolution_results/oneat_results/gt_diamond_92/Clean_CSV/' 
categories_json = '/gpfsstore/rech/jsy/uzj81mi/Mari_Models/Oneat/Cellsplitdiamondcategoriesxenopus.json'
fileextension = '*tif'
event_threshold = 0.999
nms_space = 10
nms_time = 2
Vizdetections = NEATViz(imagedir,
                        csvdir,
                        savedir,
                        categories_json,
                        segimagedir = segimagedir,
                        fileextension = fileextension,
                        volume = True,
                        event_threshold = event_threshold,
                        nms_space = nms_space,
                        nms_time = nms_time)






