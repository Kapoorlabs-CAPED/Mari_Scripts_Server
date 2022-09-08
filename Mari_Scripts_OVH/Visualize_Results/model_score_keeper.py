from oneat.NEATUtils.oneat_animation.OneatScoreKeeper import ScoreModels
from pathlib import Path
from tifffile import imread
segimage = imread('/mnt/WorkHorse/Mari_Data_Oneat/seg/tracks_gt_star.tif')
predictions = Path('/mnt/WorkHorse/Mari_Data_Oneat/oneat_metrics/')
groundtruth = '/mnt/WorkHorse/oneat_metrics/gt/gt_mitosis_locations.csv'
scoremodels = ScoreModels(segimage, predictions, groundtruth)
scoresheet = scoremodels.model_scorer()
print(scoresheet)