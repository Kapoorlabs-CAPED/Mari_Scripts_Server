#!/usr/bin/env python

# coding: utf-8


from slimoneat.NEATUtils import SLIMNEATViz




imagedir = '/mnt/d/jean_zay_backup/Mari_Data_Oneat/raw/'
segimagedir = '/mnt/d/jean_zay_backup/Mari_Data_Oneat/seg/'
csvdir = '/mnt/d/jean_zay_backup/Mari_Data_Oneat/oneat_results/'
categories_json = '/mnt/d/jean_zay_backup/Mari_Models/Oneat/Cellsplitcategoriesxenopus.json'
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






