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
        results[outlier_loader.name] = met_dict

    # print results
    for dataset_name in results.keys():
        print('\n',dataset_name)
        print('roc_auc: ', results[dataset_name][general_keys.DIRICHLET]['roc_auc'])
        print('pr_auc_norm: ', results[dataset_name][general_keys.DIRICHLET]['pr_auc_norm'])
        print('acc_at_percentil: ',
              results[dataset_name][general_keys.DIRICHLET]['acc_at_percentil'])
        print('max_accuracy: ',
              results[dataset_name][general_keys.DIRICHLET]['max_accuracy'])


    """
     small_ztfv7
    roc_auc:  0.8413602222222222
    pr_auc_norm:  0.8228540998792675
    acc_at_percentil:  0.6678333333333333
    max_accuracy:  0.7658333333333334
    
     small_hits
    roc_auc:  0.9877504444444444
    pr_auc_norm:  0.9830293993740258
    acc_at_percentil:  0.9558333333333333
    max_accuracy:  0.9578333333333333

    (504000, 21, 21, 3)
    Training Model
    Epoch 1, Loss: 0.6988180875778198, Acc: 71.2654800415039, Time: 00:00:53.03
    Epoch 2, Loss: 0.48770636320114136, Acc: 80.45416259765625, Time: 00:00:49.58
    Epoch 3, Loss: 0.25955823063850403, Acc: 89.94365692138672, Time: 00:00:49.69
    Total Training Time: 00:02:32.37

    (504000, 21, 21, 4)
    Training Model
    Epoch 1, Loss: 2.0106725692749023, Acc: 18.011308670043945, Time: 00:00:51.95
    Epoch 2, Loss: 1.7642428874969482, Acc: 27.453968048095703, Time: 00:00:50.73
    Epoch 3, Loss: 1.3397679328918457, Acc: 48.496826171875, Time: 00:00:50.94
    Total Training Time: 00:02:33.70
    """