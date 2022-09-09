from oneat.NEATUtils.oneat_animation.OneatScoreKeeper import ScoreModels
from pathlib import Path
from tifffile import imread

predictions = [Path('/mnt/WorkHorse/oneat_metrics/Vol_d56_f64.csv')]
groundtruth = '/mnt/WorkHorse/oneat_metrics/gt/gt_mitosis_locations.csv'
scoremodels = ScoreModels(predictions, groundtruth)
scoresheet = scoremodels.model_scorer()
print(scoresheet)
