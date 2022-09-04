#!/usr/bin/env python

# coding: utf-8


from oneat.NEATUtils import NEATViz

imagedir = '/mnt/d/jean_zay_backup/Mari_Data_Oneat/raw/gt/'
segimagedir = '/mnt/d/jean_zay_backup/Mari_Data_Oneat/seg/'
csvdir = '/mnt/d/jean_zay_backup/Mari_Data_Oneat/revolution_results/oneat_results/gt_scalarcnn/'
savedir = '/mnt/d/jean_zay_backup/Mari_Data_Oneat/revolution_results/oneat_results/gt_scalarcnn/Clean_CSV/' 
categories_json = '/mnt/d/jean_zay_backup/Mari_Models/Oneat/Cellsplitcategoriesxenopus.json'
fileextension = '*tif'
start_project_mid = 8
end_project_mid = 4
event_threshold = 0.9
nms_space = 10
nms_time = 2
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






