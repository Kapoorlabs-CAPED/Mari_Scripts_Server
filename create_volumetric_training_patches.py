
from oneat.NEATUtils import MovieCreator
from oneat.NEATUtils.utils import save_json
from oneat.NEATModels.TrainConfig import TrainConfig
from pathlib import Path
import hydra
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : OneatConfig):


        #Specify the directory containing images
        train_image_dir = config.paths_oneat.train_image_dir
        #Specify the directory contaiing csv files
        train_csv_dir = config.paths_oneat.train_csv_dir
        #Specify the directory containing the segmentations
        train_seg_image_dir = config.paths_oneat.train_seg_image_dir
        #Specify the model directory where we store the json of categories, training model and parameters
        model_dir = config.paths_oneat.model_dir
        #Directory for storing center ONEAT training data 
        train_save_dir = config.paths_oneat.train_save_dir
        Path(model_dir).mkdir(exist_ok = True)
        Path(train_save_dir).mkdir(exist_ok = True)


        #Name of the  events
        event_type_name = ["Normal", "Division"]
        #Label corresponding to event
        event_type_label = [0, 1]
        #The name appended before the CSV files
        csv_name_diff = 'ONEAT'
        size_tminus = config.params_train.size_tminus
        size_tplus = config.params_train.size_tplus
        trainshapex = config.params_train.imagex
        trainshapey = config.params_train.imagey
        trainshapez = config.params_train.imagez
        normalizeimage = config.params_train.normalizeimage
        npz_name = config.files_oneat.npz_name
        npz_val_name = config.files_oneat.npz_val_name
        crop_size = [trainshapex,trainshapey,trainshapez,size_tminus,size_tplus]


        event_position_name = ["x", "y", "z", "t", "h", "w", "d", "c"]
        event_position_label = [0, 1, 2, 3, 4, 5, 6, 7]

        dynamic_config = TrainConfig(event_type_name, event_type_label, event_position_name, event_position_label)

        dynamic_json, dynamic_cord_json = dynamic_config.to_json()

        save_json(dynamic_json, model_dir + '/' + config.trainclass.categories_json + '.json')

        save_json(dynamic_cord_json, model_dir + '/' + config.trainclass.cord_json + '.json')        




        MovieCreator.VolumeLabelDataSet(train_image_dir, 
                                    train_seg_image_dir, 
                                    train_csv_dir, 
                                    train_save_dir, 
                                    event_type_name, 
                                    event_type_label, 
                                    csv_name_diff,
                                    crop_size,
                                    normalizeimage = normalizeimage)
        
        MovieCreator.createNPZ(train_save_dir, axes = 'STZYXC', save_name = npz_name, save_name_val = npz_val_name)
        
if __name__=='__main__':
    main()        















