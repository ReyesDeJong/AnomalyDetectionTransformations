"""Module that defines miscellaneous keys to manipulate the model"""

# For fitting (keep best models in validation)
ITERATION = 'iteration'
LOSS = 'loss'
DISC_LOSS = 'disc_loss'

# Names of metrics
ACCURACY = 'accuracy'
IOU_ENDO = 'iou_endo'
IOU_EPI = 'iou_epi'
IOU_MYO = 'iou_myo'
DICE_ENDO = 'dice_endo'
DICE_EPI = 'dice_epi'
DICE_MYO = 'dice_myo'

# To evaluate the metrics
BATCHES_MEAN = 'batch_mean'
VALUES_PER_BATCH = 'values_per_batch'
MEAN = 'mean'
STD = 'std'

#set names
TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'

#Types of data
REAL = 'real'
GENERATED = 'generated'
FAKE_IMAGE = 'fake_image'
REAL_IMAGE = 'real_image'

#summary keys
MERGED_IMAGE_SUMM = 'merged_image_summ'

#types of variable in data
CLASS = 'class'
MAGNITUDE = 'magnitude'
TIME = 'time'
ORIGINAL_MAGNITUDE_RANDOM = 'original_magnitude_random'
TIME_RANDOM = 'time_random'
GENERATED_MAGNITUDE = 'generated_magnitude'
IMAGES = 'images'
LABELS = 'labels'