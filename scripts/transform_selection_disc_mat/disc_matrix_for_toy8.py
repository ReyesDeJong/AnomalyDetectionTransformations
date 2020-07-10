"""
Seeing if transforms are discardable in matrix space
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)
import tensorflow as tf

from models.transformer_ensemble_ovo_simple_net_od import \
    EnsembleOVOTransformODSimpleModel
from scripts.transform_selection_disc_mat.training_transform_selection import \
    get_transform_selection_transformer
from modules.trainer import ODTrainer
from parameters import param_keys

if __name__ == '__main__':
    N_TRAIN_TIMES = 10
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # GETTING MATRICES
    from parameters import loader_keys, general_keys
    from modules.geometric_transform.transformer_for_ranking import \
        RankingTransformer
    from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
    from models.transformer_od_simple_net import TransformODModel
    from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader

    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 10000,
        loader_keys.TEST_PERCENTAGE: 0.2,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.1,
        loader_keys.USED_CHANNELS: [0, 1, 2, 3],  #
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_loader = HiTSOutlierLoader(hits_params)
    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH,
            '../datasets/ALeRCE_data/new_small_od_dataset_tuples.pkl'),
    }
    ztf_loader = ZTFSmallOutlierLoader(ztf_params)
    data_loader = ztf_loader

    transformer = RankingTransformer()
    original_trf = RankingTransformer()
    print(transformer.transformation_tuples)
    print(transformer.n_transforms)

    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_loader.get_outlier_detection_datasets()
    x_train_shape = x_train.shape[1:]
    del x_train, x_val, x_test, y_train, y_val, y_test
    mdl = EnsembleOVOTransformODSimpleModel(
        data_loader=data_loader, transformer=transformer,
        input_shape=x_train_shape)
    transformer = get_transform_selection_transformer(data_loader, mdl,
                                                      transformer)
    del mdl

    print(transformer.transformation_tuples)
    print(transformer.n_transforms)
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_loader.get_outlier_detection_datasets()
    trainer = ODTrainer({
        param_keys.EPOCHS: 1000
    })
    trainer.train_and_evaluate_model_n_times(TransformODModel, transformer,
                                             x_train, x_val, x_test, y_test,
                                             N_TRAIN_TIMES, True, data_loader)
    trainer.print_metric_mean_and_std()
    trainer.train_and_evaluate_model_n_times(TransformODModel, original_trf,
                                             x_train, x_val, x_test, y_test,
                                             N_TRAIN_TIMES, True, data_loader)
    trainer.print_metric_mean_and_std()
