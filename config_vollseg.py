from dataclasses import dataclass 


@dataclass
class Params:
        depth: int 
        epochs: int 
        learning_rate: float
        batch_size: int 
        patch_x: int 
        patch_y: int 
        patch_z: int 
        kern_size: int 
        n_patches_per_image: int 
        n_rays: int 
        startfilter: int 
        use_gpu_opencl: bool 
        generate_npz: bool 
        backbone: str 
        load_data_sequence: bool 
        validation_split: float 
        n_channel_in: int 
        train_unet: bool 
        train_star: bool 
        train_loss: str
    
@dataclass
class Paths: 
    star_model_dir:  str
    unet_model_dir:  str
    roi_model_dir:  str
    den_model_dir:  str
@dataclass
class Files:  
    npz_filename: str
    model_name: str       