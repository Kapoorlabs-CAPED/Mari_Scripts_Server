import os
from pathlib import Path
import concurrent
from tifffile import imread, imwrite
import numpy as np
from oneat.NEATUtils.helpers import  normalizeFloatZeroOne
from dask.array.image import imread as daskread
inputdir = ("/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_diamond_raw/")
outputdir = "/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Training/oneat_training/oneat_train_diamond_raw_normalized/"
Path(outputdir).mkdir(exist_ok=True)
pattern = '*.tif'

files = list(inputdir.glob(pattern))
nthreads = os.cpu_count()
N = 4

def normalizer(image, i, N):
    smallimage = image[i * image.shape[0] // N:(i + 1) * image.shape[0]//N,:] 
    newimage =  normalizeFloatZeroOne( smallimage,1,99.8, dtype= np.float16)
    print(i)
    return newimage , i  

for fname in files:
    with concurrent.futures.ThreadPoolExecutor(max_workers = nthreads) as executor:
     futures = []
     
     image = daskread(fname)[0].astype('float16')
     newimage = image
     name = os.path.splitext(os.path.basename(fname)[0])
     for i in range(N):

        futures.append(executor.submit(image, i, N))
     for future in concurrent.futures.as_completed(futures):
            returnimage, index = future.result()
            newimage[index * image.shape[0] // N:(index + 1) * image.shape[0]//N,:] = returnimage
     imwrite(outputdir + '/' + os.path.splitext(name)[0] + '.tif', newimage)
