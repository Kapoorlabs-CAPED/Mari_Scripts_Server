from dataclasses import dataclass

@dataclass
class Params:
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
    
@dataclass
class Files:
    npz_name: str
    npz_val_name: str
    model_name: str
    vollnet_categories_json: str
    vollnet_cord_json: str
    lstmnet_categories_json: str
    lstmnet_cord_json: str
    
@dataclass
class Paths:
    model_dir : str  
    npz_directory : str       
    
@dataclass 
class TrainOneatConfig:
    
    paths: Paths 
    files: Files 
    params: Params    