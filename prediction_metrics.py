import hydra
from caped_ai_metrics import ClassificationScore
from config_oneat import OneatConfig
from hydra.core.config_store import ConfigStore

configstore = ConfigStore.instance()
configstore.store(name = 'OneatConfig', node = OneatConfig)

@hydra.main(config_path = 'conf', config_name = 'config_oneat')
def main(config : OneatConfig):
    
    seg_image_dir = config.paths_oneat.metrics_seg_dir
    gt_seg_image = config.files_oneat.gt_seg_image

    predictions_dir = config.paths_oneat.metrics_supress_csv_dir
    gt_csv_dir = config.paths_oneat.metrics_gt_csv_dir
    gt_csv_file = config.files_oneat.gt_csv   
    
    groundtruth = gt_csv_dir + gt_csv_file

    
    thresholdscore = 0.999
    compute_score = ClassificationScore(predictions_dir, groundtruth, thresholdspace = 40, thresholdtime = 4, thresholdscore = thresholdscore, metric = 'Euclid')
    compute_score.model_scorer()

if __name__=='__main__':
    main()    
