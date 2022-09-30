from oneat.NEATUtils.oneat_animation.OneatScoreKeeper import ScoreModels
from pathlib import Path
from tifffile import imread

predictions = [Path('/mnt/WorkHorse/oneat_metrics/CleanCSV/CleanSca_d56_f64.csv'),Path('/mnt/WorkHorse/oneat_metrics/CleanCSV/CleanSca_d101_f64.csv'), Path('/mnt/WorkHorse/oneat_metrics/CleanCSV/CleanSca_d56_f48.csv'),  Path('/mnt/WorkHorse/oneat_metrics/CleanCSV/CleanLstm_d38_f48.csv'), Path('/mnt/WorkHorse/oneat_metrics/CleanCSV/CleanLstm_d56_f48.csv'), Path('/mnt/WorkHorse/oneat_metrics/CleanCSV/CleanLstm_d56_f64.csv'), Path('/mnt/WorkHorse/oneat_metrics/CleanCSV/CleanVol_d101_f64.csv'), Path('/mnt/WorkHorse/oneat_metrics/CleanCSV/CleanVol_d56_f64.csv'), Path('/mnt/WorkHorse/oneat_metrics/CleanCSV/CleanVol_d101_f48.csv') ]
groundtruth = '/mnt/WorkHorse/oneat_metrics/gt/gt_mitosis_locations.csv'
thresholdscore = 0.99999
thresholdspace = 40
thresholdtime = 4
scoremodels = ScoreModels(predictions, groundtruth, thresholdscore = thresholdscore, thresholdspace = thresholdspace, thresholdtime = thresholdtime)
scoresheet = scoremodels.model_scorer()
print(scoresheet)
