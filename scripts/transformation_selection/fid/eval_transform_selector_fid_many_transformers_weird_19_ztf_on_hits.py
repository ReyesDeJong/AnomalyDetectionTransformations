"""
Training a model with basic-non-composed transforms, to visualize it feature
 layer and
then calculate FID
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)

from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from modules.data_loaders.ztf_outlier_loader import ZTFOutlierLoader
from modules import utils
from modules.transform_selection.fid_modules.transform_selector_fid import \
    TransformSelectorFRawLogFID
from parameters import loader_keys, general_keys, param_keys
from modules.geometric_transform.transformer_for_ranking import \
    RankingTransformer
from modules.data_loaders.ztf_small_outlier_loader import ZTFSmallOutlierLoader
from modules.geometric_transform.transformations_tf import Transformer, \
    PlusKernelTransformer
from models.transformer_od_simple_net import TransformODModel
from modules.trainer import ODTrainer

if __name__ == "__main__":
    N_TRAIN_TIMES = 10

    utils.init_gpu_soft_growth()
    # data loaders
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '../datasets/HiTS2013_300k_samples.pkl'),
        loader_keys.N_SAMPLES_BY_CLASS: 100000,
        loader_keys.TEST_PERCENTAGE: 0.0,
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
        loader_keys.USED_CHANNELS: [2],  # [0, 1, 2, 3],  #
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    hits_params.update({loader_keys.USED_CHANNELS: [0, 1, 2, 3]})
    hits_loader_4c = HiTSOutlierLoader(hits_params, pickles_usage=False)
    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH,
            '../datasets/ALeRCE_data/converted_pancho_septiembre.pkl'),
        loader_keys.VAL_SET_INLIER_PERCENTAGE: 0.0,
        loader_keys.USED_CHANNELS: [0, 1, 2],
        loader_keys.CROP_SIZE: 21,
        general_keys.RANDOM_SEED: 42,
        loader_keys.TRANSFORMATION_INLIER_CLASS_VALUE: 1
    }
    ztf_loader_3c = ZTFOutlierLoader(ztf_params, pickles_usage=False)

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

    data_loaders_both = [
        # (hits_loader, hits_loader_4c),
        # (ztf_loader, ztf_loader_3c)
        (hits_loader, ztf_loader_3c)
    ]

    transformer_rank = RankingTransformer()
    transformer_72 = Transformer()
    transformer_99 = PlusKernelTransformer()

    transformers = [
        transformer_rank,
        transformer_72,
        transformer_99
    ]

    fid_selector = TransformSelectorFRawLogFID()

    f = open('eval_fid_weird_19_of_ztf_on_hits.txt', 'w')

    for data_loaders_both_i in data_loaders_both:
        data_loader_to_train = data_loaders_both_i[0]
        data_loader_to_fid = data_loaders_both_i[1]
        print('\nDataLoader Name %s ' % data_loader_to_train.name, file=f, flush=True)
        for transformer_i in transformers:
            orig_trfs = transformer_i.transformation_tuples[:]
            print('\nInit Trf Name %s %i' % (
            transformer_i.name, len(transformer_i.transformation_tuples)),
                  file=f, flush=True)
            x_train_fid = \
            data_loader_to_fid.get_outlier_detection_datasets()[0][0]
            (x_train, y_train), (x_val, y_val), (
                x_test,
                y_test) = data_loader_to_train.get_outlier_detection_datasets()
            trainer = ODTrainer({
                param_keys.EPOCHS: 1000
            })
            selected_trfs = fid_selector.get_selected_transformations(
                x_train_fid, transformer_i, verbose=True)
            transformer_i.set_transformations_to_perform(selected_trfs)
            print(
                'Selected Trf %i %s' % (len(selected_trfs), str(selected_trfs)),
                file=f, flush=True)
            trainer.train_and_evaluate_model_n_times(
                TransformODModel, transformer_i, x_train, x_val, x_test, y_test,
                N_TRAIN_TIMES, True, data_loader_to_train)
            mean, std = trainer.get_metric_mean_and_std()
            msg = "%i TIMES RESULTS %s n_trf%i %s : %.4f +/- %.4f" % \
                  (N_TRAIN_TIMES, data_loader_to_train.name,
                   transformer_i.n_transforms,
                   trainer.score_name,
                   mean,
                   std)
            print(msg, file=f, flush=True)
            transformer_i.set_transformations_to_perform(orig_trfs)
            print('\nFinal Trf Name %s %i' % (
                transformer_i.name, len(transformer_i.transformation_tuples)),
                  file=f, flush=True)
    f.close()
