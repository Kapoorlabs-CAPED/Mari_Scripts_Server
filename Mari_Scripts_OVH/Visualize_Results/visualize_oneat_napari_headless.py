#!/usr/bin/env python

# coding: utf-8


from slimoneat.NEATUtils import SLIMNEATViz




imagedir = '/mnt/WorkHorse/Mari_Data_Oneat/raw/'
segimagedir = '/mnt/WorkHorse/Mari_Data_Oneat/seg/'
csvdir = '/mnt/WorkHorse/Mari_Data_Oneat/oneat_results/'
categories_json = '/mnt/WorkHorse/Mari_Models/Oneat/Cellsplitcategoriesxenopus.json'
fileextension = '*tif'
start_project_mid = 4
end_project_mid = 1
Vizdetections = SLIMNEATViz(imagedir,
                        csvdir,
                        categories_json,
                        segimagedir = segimagedir,
                        fileextension = fileextension,
                        start_project_mid = start_project_mid,
                        end_project_mid = end_project_mid)






