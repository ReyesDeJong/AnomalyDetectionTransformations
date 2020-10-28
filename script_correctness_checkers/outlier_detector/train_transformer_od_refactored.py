"""
First attempt (train_step_tf2 like) to build geometric trasnformer for outlier detection in tf2
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)
from models.geotransform_wrn import GeoTransformWRN
import time
from modules.data_loaders.ztf_outlier_loaderv2 import ZTFOutlierLoaderv2
from modules.data_loaders.hits_outlier_loaderv2 import HiTSOutlierLoaderv2
from parameters import loader_keys, general_keys
from modules.geometric_transform import transformations_tf
from modules import utils

if __name__ == '__main__':
    utils.set_soft_gpu_memory_growth()
    # data loaders
    #ztf
    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '..', 'datasets', 'thesis_data', 'ztfv7',
            'preprocessed_21', 'ztf_small_dict.pkl'),
    }
    ztf_loader = ZTFOutlierLoaderv2(ztf_params, 'small_ztfv7')
    #hits
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '..', 'datasets', 'thesis_data', 'hits',
            'hits_small_4c_tuples.pkl'),
    }
    hits_loader = HiTSOutlierLoaderv2(hits_params, 'small_hits')
    data_loaders = [ztf_loader, hits_loader]
    results = {}
    for outlier_loader in data_loaders:
        (x_train, y_train), (x_val, y_val), (
            x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
        # transformer
        transformer = transformations_tf.Transformer()#TransTransformer()
        print('n transforms: ', transformer.n_transforms)
        # model
        model = GeoTransformWRN(
            transformer, results_folder_name='Single_run',
            name='GeoTransform_WRN_%s' % outlier_loader.name)
        model.fit(x_train, x_val=None, epochs=None)
        model.update_prediction_threshold(x_val)
        # model evaluation
        start_time = time.time()
        met_dict = model.evaluate(
            x_test, y_test, outlier_loader.name, 'real')
        print(
            "Time model.evaluate %s" % utils.timer(
                start_time, time.time()),
            flush=True)
        results[outlier_loader.name] = met_dict

    # print results
    for dataset_name in results.keys():
        print('\n',dataset_name)
        print('roc_auc: ', results[dataset_name]['roc_auc'])
        print('pr_auc_norm: ', results[dataset_name]['pr_auc_norm'])
        print('acc_at_percentil: ',
              results[dataset_name]['acc_at_percentil'])
        print('max_accuracy: ',
              results[dataset_name]['max_accuracy'])


    """
     small_ztfv7
    roc_auc:  0.8721593333333333
    pr_auc_norm:  0.8596773563389756
    acc_at_percentil:  0.6996666666666667
    max_accuracy:  0.7985
    
     small_hits
    roc_auc:  0.9827832222222223
    pr_auc_norm:  0.974551065334756
    acc_at_percentil:  0.95
    max_accuracy:  0.9508333333333333

    (504000, 21, 21, 3)
    Training Model
    Time usage: 00:00:53.44
    (Train dataset) Epoch 1 Iteration 3936: loss 0.827485, acc 66.223816
    Time usage: 00:01:43.29
    (Train dataset) Epoch 2 Iteration 7873: loss 0.469407, acc 80.932541
    Time usage: 00:02:33.36
    (Train dataset) Epoch 3 Iteration 11810: loss 0.291184, acc 88.768456
    Total training time: 00:02:33.39

    (504000, 21, 21, 4)
    Training Model
    Time usage: 00:00:52.54
    (Train dataset) Epoch 1 Iteration 3936: loss 2.025093, acc 17.476389
    Time usage: 00:01:43.49
    (Train dataset) Epoch 2 Iteration 7873: loss 1.750357, acc 27.340874
    Time usage: 00:02:34.45
    (Train dataset) Epoch 3 Iteration 11810: loss 1.299774, acc 48.986111
    
    Total training time: 00:02:34.48
    """