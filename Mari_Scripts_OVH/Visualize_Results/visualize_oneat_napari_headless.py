#!/usr/bin/env python

# coding: utf-8


from oneat.NEATUtils import NEATViz




imagedir = '/mnt/WorkHorse/Mari_Data_Oneat/raw/gt/'
segimagedir = '/mnt/WorkHorse/Mari_Data_Oneat/seg/'
csvdir = '/mnt/WorkHorse/oneat_metrics/'
savedir = '/mnt/WorkHorse/oneat_metrics/CleanCSV/'
categories_json = '/mnt/WorkHorse/Mari_Models/Oneat/Cellsplitcategoriesxenopus.json'
fileextension = '*tif'
start_project_mid = 4
end_project_mid = 1
Vizdetections = NEATViz(imagedir,
                        csvdir,
                        savedir,
                        categories_json,
                        segimagedir = segimagedir,
                        headless = True,
                        batch = True,
                        fileextension = fileextension,
                        start_project_mid = start_project_mid,
                        end_project_mid = end_project_mid)






