import os
import glob
from tifffile import imread
from vollseg import StarDist3D, UNET, VollCellSeg, MASKUNET

import hydra
from config_vollseg import VollSegConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'VollSegConfig', node = VollSegConfig)

@hydra.main(config_path = 'conf', config_name = 'config_vollseg')
def main( config : VollSegConfig):

                image_dir = config.paths_vollseg.predict_image_dir
                model_dir = config.paths_vollseg.model_dir
                save_dir = config.paths_vollseg.save_dir
                
                unet_model_name = config.files_vollseg.unet_model_name
                star_model_name = config.files_vollseg.star_model_name
                roi_model_name = config.files_vollseg.roi_model_name
                pretrained_cellpose_model_path = config.paths_vollseg.cellpose_model_dir
                cellpose_model = pretrained_cellpose_model_path
                custom_cellpose_model = config.params.custom_cellpose_model
                cellpose_model_name = config.files_vollseg.cellpose_model_name
                diameter_cellpose = config.params.diameter_cellpose
                stitch_threshold = config.params.stitch_threshold
                channel_membrane = config.params.channel_membrane
                channel_nuclei = config.params.channel_nuclei
                flow_threshold = config.params.flow_threshold
                cellprob_threshold = config.params.cellprob_threshold
                anisotropy = config.params.anisotropy
                
                
                unet_model = UNET(config = None, name = unet_model_name, basedir = model_dir)
                star_model = StarDist3D(config = None, name = star_model_name, basedir = model_dir)
                roi_model =  MASKUNET(config = None, name = roi_model_name, basedir = model_dir)
            


                Raw_path = os.path.join(image_dir, config.params.file_type)
                filesRaw = glob.glob(Raw_path)
                filesRaw.sort
               
                min_size = config.params.min_size
                min_size_mask = config.params.min_size_mask
                max_size = config.params.max_size
                n_tiles = config.params.n_tiles
                dounet = config.params.dounet
                seedpool = config.params.seedpool
                slice_merge = config.params.slice_merge
                UseProbability = config.params.UseProbability
                donormalize = config.params.donormalize
                axes = config.params.axes
                ExpandLabels = config.params.ExpandLabels
                for fname in filesRaw:
                
                                image = imread(fname)
                                Name = os.path.basename(os.path.splitext(fname)[0])
                                VollCellSeg( image, 
                                        diameter_cellpose = diameter_cellpose,
                                        stitch_threshold = stitch_threshold,
                                        channel_membrane = channel_membrane,
                                        channel_nuclei = channel_nuclei,
                                        flow_threshold = flow_threshold, 
                                        cellprob_threshold = cellprob_threshold,    
                                        unet_model = unet_model, 
                                        star_model = star_model, 
                                        roi_model= roi_model,
                                        seedpool = seedpool, 
                                        cellpose_model = cellpose_model, 
                                        custom_cellpose_model = custom_cellpose_model, 
                                        pretrained_cellpose_model_path = pretrained_cellpose_model_path,
                                        cellpose_model_name = cellpose_model_name,
                                        anisotropy = anisotropy,
                                        axes = axes, 
                                        min_size = min_size,  
                                        min_size_mask = min_size_mask,
                                        max_size = max_size,
                                        donormalize=donormalize,
                                        n_tiles = n_tiles,
                                        ExpandLabels = ExpandLabels,
                                        slice_merge = slice_merge, 
                                        UseProbability = UseProbability, 
                                        save_dir = save_dir, 
                                        Name = Name,
                                        dounet = dounet)   
                                        
if __name__=='__main__':
     main()                                         
