from config_oneat import TrainOneatConfig
import hydra
from hydra.core.config_store import ConfigStore


configstore = ConfigStore.instance()
configstore.store(name = 'TrainOneatConfig', node = TrainOneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main( config : TrainOneatConfig):
    
    print(config.trainclass)
    
    
if __name__== '__main__':
    main()    