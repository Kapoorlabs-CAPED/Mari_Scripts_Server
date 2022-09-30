from tifffile import imread, imwrite
from dask.array.image import imread as daskread
from pathlib import Path
import numpy as np
import os
import glob
from natsort import natsorted
imagedir = '/mnt/WorkHorse/Mari_Data_Oneat/seg/second_dataset/'
savedir = '/mnt/WorkHorse/Mari_Data_Oneat/seg/second_dataset/split_dataset/'
Path(savedir).mkdir(exist_ok=True)
Raw_path = os.path.join(imagedir, '*tif')
filesRaw = glob.glob(Raw_path)
filesRaw = natsorted(filesRaw)
dtype = np.uint16
for imagename in filesRaw:
                print(imagename)
                #image = imread(imagename).astype(dtype)
                image = daskread(imagename)
                print('image read') 
                Name = os.path.basename(os.path.splitext(imagename)[0])
                
                
                imwrite(savedir + '/' + Name + str(240)  +  '.tif', np.array(image[0,220:240,:,:,:]) )
                imwrite(savedir + '/' + Name + str(260)  +  '.tif', np.array(image[0,240:260,:,:,:]) )
                
