import os
from pathlib import Path
import concurrent
from tifffile import imread, imwrite
import numpy as np
from oneat.NEATUtils.helpers import  normalizeFloatZeroOne

inputdir = Path("/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_diamond_raw/")
outputdir = "/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_diamond_raw_normalized/"
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.tif'

files = list(inputdir.glob(pattern))
nthreads = 1 
#os.cpu_count()
def normalizer(file):
    image = imread(file)
    newimage =  normalizeFloatZeroOne( image.astype('float32'),1,99.8, dtype= np.float16)
    return newimage, file.name   
with concurrent.futures.ThreadPoolExecutor(max_workers = nthreads) as executor:
     futures = []
     for fname in files:
         futures.append(executor.submit(normalizer, file = fname))
     for future in concurrent.futures.as_completed(futures):
                   newimage, name = future.result()
                   imwrite(outputdir + '/' + os.path.splitext(name)[0] + '.tif', newimage)
