from oneat.NEATModels.loss import volume_yolo_loss
from oneat.NEATModels.nets import Concat
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
import os
from oneat.NEATUtils.utils import load_json

pb_model_dir = '/home/debian/WorkHorse/Mari_Models/Oneat/oneat_xenopus_volumetric/'
h5_model = '/home/debian/WorkHorse/Mari_Models/Oneat/oneat_xenopus_volumetric/weights.h5'



division_categories_json = pb_model_dir + 'catagories.json'
catconfig = load_json(division_categories_json)
config = load_json(os.path.join(pb_model_dir, 'parameters.json'))

            
box_vector = config['box_vector']
categories = len(catconfig)
depth = config['depth']
start_kernel = config['start_kernel']
mid_kernel = config['mid_kernel']
imagex = config['imagex']
imagey = config['imagey']
imagez = config['imagez']
imaget = config['size_tminus'] + config['size_tplus'] + 1
size_tminus = config['size_tminus']
size_tplus = config['size_tplus']
nboxes = config['nboxes']
stage_number = config['stage_number']
last_conv_factor = 2 ** (stage_number - 1)
gridx = 1
gridy = 1
gridz = 1
entropy = 'notbinary'
yololoss = volume_yolo_loss(categories, gridx, gridy, gridz, nboxes,box_vector, entropy)
with tensorflow.device('/cpu:0'):
   
     model = load_model(pb_model_dir,
                                custom_objects={'loss': yololoss, 'Concat': Concat})
     
save_model(model, h5_model)     
