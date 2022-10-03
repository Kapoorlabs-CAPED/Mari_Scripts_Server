#!/usr/bin/env python
# coding: utf-8


from vollseg import SmartSeeds3D
import hydra
from config_vollseg import VollSegConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'VollSegConfig', node = VollSegConfig)

@hydra.main(config_path = 'conf', config_name = 'config_vollseg')
def main( config : VollSegConfig):
        base_dir = config.paths_vollseg.base_dir
        npz_filename = config.files_vollseg.npz_filename
        model_dir = config.paths_vollseg.model_dir
        model_name = config.files_vollseg.model_name

        raw_dir = config.paths_vollseg.raw_dir
        real_mask_dir = config.paths_vollseg.real_mask_dir
        binary_mask_dir = config.paths_vollseg.binary_mask_dir

        #Network training parameters
        depth = config.params.depth
        epochs = config.params.epochs
        learning_rate = config.params.learning_rate
        batch_size = config.params.batch_size
        patch_x = config.params.patch_x
        patch_y = config.params.patch_y
        patch_z = config.params.patch_z
        kern_size = config.params.kern_size
        n_patches_per_image = config.params.n_patches_per_image
        n_rays = config.params.n_rays
        erosion_iterations = config.params.erosion_iterations
        startfilter = config.params.startfilter
        use_gpu_opencl = config.params.use_gpu_opencl
        generate_npz = config.params.generate_npz
        backbone = config.params.backbone
        load_data_sequence = config.params.load_data_sequence
        validation_split = config.params.validation_split
        n_channel_in = config.params.n_channel_in
        train_unet = config.params.train_unet
        train_star = config.params.train_star
        train_loss = config.params.train_loss



        SmartSeeds3D(base_dir = base_dir, 
                    npz_filename = npz_filename, 
                    model_name = model_name, 
                    model_dir = model_dir,
                    raw_dir = raw_dir,
                    real_mask_dir = real_mask_dir,
                    binary_mask_dir = binary_mask_dir,
                    n_channel_in = n_channel_in,
                    backbone = backbone,
                    load_data_sequence = load_data_sequence, 
                    validation_split = validation_split, 
                    n_patches_per_image = n_patches_per_image, 
                    generate_npz = generate_npz,
                    patch_x= patch_x, 
                    patch_y= patch_y, 
                    patch_z = patch_z,
                    erosion_iterations = erosion_iterations,  
                    train_loss = train_loss,
                    train_star = train_star,
                    train_unet = train_unet,
                    use_gpu = use_gpu_opencl,  
                    batch_size = batch_size, 
                    depth = depth, 
                    kern_size = kern_size, 
                    startfilter = startfilter, 
                    n_rays = n_rays, 
                    epochs = epochs, 
                    learning_rate = learning_rate)
   
if __name__ == '__main__':
    main()        

