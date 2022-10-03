from dataclasses import dataclass

@dataclass
class ParamsTrain:
    startfilter: int
    start_kernel: int 
    mid_kernel: int 
    depth: int 
    learning_rate: float 
    batch_size: int 
    epochs: int 
    show: bool 
    stage_number: int 
    size_tminus: int 
    size_tplus: int 
    imagex: int 
    imagey: int
    imagez: int
    nboxes: int
    pure_lstm: bool
    normalizeimage: bool



@dataclass 
class ParamsPredict:
     n_tiles: tuple
     event_threshold: float
     event_confidence: float 
     downsamplefactor: int 
     start_project_mid: int 
     end_project_mid: int 
     normalize: bool 
     nms_function: str 
     file_type: str
     remove_markers: bool
     nms_space: int 
     nms_time: int
     volume: bool
     
     
    
@dataclass
class Trainclass:
    training_class : type    
    training_config : type
    
@dataclass
class Files:
    npz_name: str
    npz_val_name: str
    model_name: str
    categories_json: str
    voll_cord_json: str
    lstm_cord_json: str
    
@dataclass
class Paths:
    model_dir : str  
    npz_directory : str   
    imagedir: str
    segdir: str
    savedir: str 
    train_image_dir: str
    train_seg_image_dir: str
    train_csv_dir: str
    train_save_dir: str  
        
    
@dataclass 
class OneatConfig:
    
    paths_oneat: Paths 
    files_oneat: Files 
    params_train: ParamsTrain 
    params_predict: ParamsPredict 
    trainclass :Trainclass 
      