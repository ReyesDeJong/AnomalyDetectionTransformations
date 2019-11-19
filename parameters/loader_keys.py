"""Defines the parameters to configure the data loaders."""

"""
Oulier detection loader params

validation_random_seed: (int) Random seed to be used to split the validation
    data from the training data.
"""
VAL_SET_INLIER_PERCENTAGE = 'val_percentage_of_inliers'
USED_CHANNELS = 'used_channels'
DATA_PATH = 'data_path'
CROP_SIZE = 'crop_size'
TRANSFORMATION_INLIER_CLASS_VALUE = 'transformation_outlier_class_value'
ZTF_OUTLIER_CLASS_VALUE = 'ztf_outlier_class_value'