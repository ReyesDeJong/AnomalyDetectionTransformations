"""
First attempt (train_step_tf2 like) to build geometric trasnformer for outlier detection in tf2
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)
from models.transformer_od import TransformODModel
import time
from modules.data_loaders.ztf_outlier_loaderv2 import ZTFOutlierLoaderv2
from modules.data_loaders.hits_outlier_loaderv2 import HiTSOutlierLoaderv2
from parameters import loader_keys
from modules.geometric_transform import transformations_tf
from modules import utils

if __name__ == '__main__':
    utils.set_soft_gpu_memory_growth()
    # data loader
    # ztf_params = {
    #     loader_keys.DATA_PATH: os.path.join(
    #         PROJECT_PATH, '..', 'datasets', 'thesis_data', 'ztfv7',
    #         'preprocessed_21', 'ztf_small_dict.pkl'),
    # }
    # outlier_loader = ZTFOutlierLoaderv2(ztf_params, 'small_ztfv7')
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '..', 'datasets', 'thesis_data', 'hits',
            'hits_small_4c_tuples.pkl'),
    }
    outlier_loader = HiTSOutlierLoaderv2(hits_params, 'small_hits')
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
    # transformer
    transformer = transformations_tf.Transformer()#TransTransformer()
    print('n transforms: ', transformer.n_transforms)
    # model
    model = TransformODModel(
        data_loader=outlier_loader, transformer=transformer,
        input_shape=x_train.shape[1:])
    model.fit(x_train, x_val=None, epochs=None)
    # model evaluation
    start_time = time.time()
    met_dict = model.evaluate_od(
        x_train, x_test, y_test, 'ztf_small_v7-real-bog', 'real', x_val)
    print(
        "Time model.evaluate_od %s" % utils.timer(
            start_time, time.time()),
        flush=True)
    print('\nroc_auc')
    for key in met_dict.keys():
        print(key, met_dict[key]['roc_auc'])
    print('\nacc_at_percentil')
    for key in met_dict.keys():
        print(key, met_dict[key]['acc_at_percentil'])
    print('\nmax_accuracy')
    for key in met_dict.keys():
        print(key, met_dict[key]['max_accuracy'])
