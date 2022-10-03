#!/usr/bin/env python
# coding: utf-8

# In[1]:


import napari
import os
import glob
from pathlib import Path
from tifffile import imread
import pandas as pd
from qtpy.QtWidgets import QComboBox, QPushButton
from qtpy.QtWidgets import QApplication
app = QApplication([])




import hydra
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)


class EventViewer(object):
    
    def __init__(self, viewer, imagename, Name, csv_dir, save = False, newimage = True, start_project_mid = 4, end_project_mid = 1):
     
          self.save = save
          self.newimage = newimage
          self.viewer = viewer
          self.start_project_mid = start_project_mid
          self.end_project_mid = end_project_mid  
          print('reading image')      
          self.imagename = imagename  
          self.image = imread(imagename)
          print('image read')
          self.Name = Name
          self.ndim = len(self.image.shape)  
          self.csv_dir = csv_dir
          
          
          self.click()
         
    def click(self):

        
                ClassBackground = 'Normal'
                ClassDivision = 'Division'
                
                if self.save == True:
 
                        ClassBackgrounddata = self.viewer.layers[ClassBackground].data
                        bgdf = pd.DataFrame(ClassBackgrounddata, index = None, columns = ['T', 'Z', 'Y', 'X'])
                        bgdf.to_csv(self.csv_dir + '/' + 'ONEAT' + ClassBackground + self.Name +  '.csv', index = False, mode = 'w')

                        ClassDivisiondata = self.viewer.layers[ClassDivision].data
                        divdf = pd.DataFrame(ClassDivisiondata, index = None, columns = ['T', 'Z', 'Y', 'X'])
                        divdf.to_csv(self.csv_dir + '/' +'ONEAT' + ClassDivision + self.Name +  '.csv', index = False, mode = 'w')

                       
        
                if self.newimage == True:
                     for layer in list(self.viewer.layers):

                            self.viewer.layers.remove(layer) 
                    
                if self.save == False:
                        self.viewer.add_image(self.image, name = self.Name)


                        # add the first points layer for ClassBackground T point
                        self.viewer.add_points(name= ClassBackground, face_color='red', ndim = 4)

                        # add the second points layer for ClassDivision T point
                        self.viewer.add_points(name= ClassDivision, face_color='blue', ndim = 4)

                      
                        # programatically enter add mode for both Points layers to enable editing

                        self.viewer.layers[ClassBackground].mode = 'add'
                        self.viewer.layers[ClassDivision].mode = 'add'
                       

                

def main(config : OneatConfig):

   sourcedir = config.paths_oneat.train_basic_image_dir
   csv_dir = config.paths_oneat.train_basic_csv_dir
   Imageids = []
   Boxname = 'ImageIDBox'
   Path(csv_dir).mkdir(exist_ok = True)



   Raw_path = os.path.join(sourcedir, config.params_predict.file_type)
   X = glob.glob(Raw_path)
   for imagename in X:
         Imageids.append(imagename)
         Name = os.path.basename(os.path.splitext(imagename)[0])

   imageidbox = QComboBox()   
   imageidbox.addItem(Boxname)   
   tracksavebutton = QPushButton('Save Clicks')

   for i in range(0, len(Imageids)):


       imageidbox.addItem(str(Imageids[i]))
        
        
   viewer = napari.Viewer()        
   viewer.window.add_dock_widget(imageidbox, name="Image", area='bottom')    
   viewer.window.add_dock_widget(tracksavebutton, name="Save Clicks", area='bottom')
   imageidbox.currentIndexChanged.connect(
         lambda trackid = imageidbox: EventViewer(
                 viewer,
                  imageidbox.currentText(),
                       os.path.basename(os.path.splitext(imageidbox.currentText())[0]), csv_dir, False, True ))     

   tracksavebutton.clicked.connect(
        lambda trackid= tracksavebutton:EventViewer(
                 viewer,
                  imageidbox.currentText(),
                       os.path.basename(os.path.splitext(imageidbox.currentText())[0]), csv_dir, True, False ))


   napari.run()


if __name__=='__main__':
     
     main()


