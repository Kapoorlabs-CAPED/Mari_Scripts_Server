import hydra
from config_vollseg import VollSegConfig
from hydra.core.config_store import ConfigStore
from vollseg import SmartPatches

configstore = ConfigStore.instance()
configstore.store(name = 'VollSegConfig', node = VollSegConfig)

@hydra.main(config_path = 'conf', config_name = 'config_vollseg')
def main( config : VollSegConfig):
        base_dir = config.paths_vollseg.base_dir

        raw_dir = config.paths_vollseg.raw_dir
        real_mask_dir = config.paths_vollseg.real_mask_dir
        binary_mask_dir = config.paths_vollseg.binary_mask_dir
        raw_save_dir = config.paths_vollseg.raw_save_dir
        real_mask_patch_dir = config.paths_vollseg.real_mask_patch_dir
        patch_size = (config.params.patch_z, config.params.patch_y, config.params.patch_x)
        erosion_iterations = config.params.erosion_iterations
        
        SmartPatches(base_dir, raw_dir, real_mask_dir, raw_save_dir, real_mask_patch_dir, binary_mask_dir, patch_size, erosion_iterations)
        
if __name__=='__main__':
    
    main()        