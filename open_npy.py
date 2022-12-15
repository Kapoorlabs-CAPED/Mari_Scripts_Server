import numpy as np
import os
from tifffile import imwrite
import hydra
from hydra.core.config_store import ConfigStore
from config_vollseg import VollSegConfig
import glob
from pathlib import Path

configstore = ConfigStore.instance()
configstore.store(name = 'VollSegConfig' , node = VollSegConfig )

@hydra.main(config_path= 'conf', config_name = 'config_vollseg')
def main(config: VollSegConfig):
    
    npy_dir = config.paths_vollseg.npy_dir 
    save_dir = config.paths_vollseg.npy_mask_dir
    Path(save_dir).mkdir(exist_ok=True)
    Raw_path = os.path.join(npy_dir, '*.npy')
    filesRaw = glob.glob(Raw_path)
    filesRaw.sort
    for fname in filesRaw:
        data = np.load(fname, allow_pickle=True).item()
        label_image = data['masks']
        imwrite(os.path.join(save_dir, fname.stem), label_image.astype('uint16'))
        
        
        


if __name__=='__main__':
     main() 