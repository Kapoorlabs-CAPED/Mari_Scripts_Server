#!/usr/bin/env python

# coding: utf-8


from oneat.NEATUtils import NEATViz




imagedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/raw/gt/'
segimagedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/seg/'
csvdir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/oneat_retrain_results/d38f32l32/'
savedir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/oneat_retrain_results/d38f32l32/Clean_CSV/' 
categories_json = '/gpfsstore/rech/jsy/uzj81mi/Mari_Models/Oneat/Cellsplitcategoriesxenopus.json'
fileextension = '*tif'
start_project_mid = 8
end_project_mid = 4
event_threshold = 0.9
nms_space = 10
nms_time = 3
Vizdetections = NEATViz(imagedir,
                        csvdir,
                        savedir,
                        categories_json,
                        segimagedir = segimagedir,
                        fileextension = fileextension,
                        start_project_mid = start_project_mid,
                        end_project_mid = end_project_mid, headless = False,
                        event_threshold = event_threshold,
                        nms_space = nms_space,
                        nms_time = nms_time)






