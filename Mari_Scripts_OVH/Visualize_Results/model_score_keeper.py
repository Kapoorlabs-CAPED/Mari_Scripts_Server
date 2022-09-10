from oneat.NEATUtils.oneat_animation.OneatScoreKeeper import ScoreModels
from pathlib import Path
from tifffile import imread

predictions = [Path('/mnt/WorkHorse/oneat_metrics/Vol_d56_f64.csv'), Path('/mnt/WorkHorse/oneat_metrics/Sca_d56_f64.csv'),Path('/mnt/WorkHorse/oneat_metrics/Sca_d56_f48.csv'), Path('/mnt/WorkHorse/oneat_metrics/Lstm_d56_f48.csv'), Path('/mnt/WorkHorse/oneat_metrics/Lstm_d56_f64.csv'), Path('/mnt/WorkHorse/oneat_metrics/Vol_d101_f48.csv'), Path('/mnt/WorkHorse/oneat_metrics/Lstm_d38_f48.csv')]
groundtruth = '/mnt/WorkHorse/oneat_metrics/gt/gt_mitosis_locations.csv'
thresholdscore = 0.9
thresholdspace = 40
thresholdtime = 16
scoremodels = ScoreModels(predictions, groundtruth, thresholdscore = thresholdscore, thresholdspace = thresholdspace, thresholdtime = thresholdtime)
scoresheet = scoremodels.model_scorer()
print(scoresheet)
