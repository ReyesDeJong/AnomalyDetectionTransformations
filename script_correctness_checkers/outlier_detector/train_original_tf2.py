"""
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)
import numpy as np
from scipy.special import psi, polygamma
from tensorflow.keras.utils import to_categorical
from modules.geometric_transform.transformations_original_tf2 import Transformer
from models.wide_residual_network_original_tf2 import \
    create_wide_residual_network
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from modules.data_loaders.ztf_outlier_loaderv2 import ZTFOutlierLoaderv2
from modules.data_loaders.hits_outlier_loaderv2 import HiTSOutlierLoaderv2
from parameters import loader_keys


def get_roc_pr_curve_data(scores, labels):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate(
        (np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(
        truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(
        truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)
    result_dict = {
        'preds': preds, 'truth': truth,
        'fpr': fpr, 'tpr': tpr, 'roc_thresholds': roc_thresholds,
        'roc_auc': roc_auc,
        'precision_norm': precision_norm, 'recall_norm': recall_norm,
        'pr_thresholds_norm': pr_thresholds_norm,
        'pr_auc_norm': pr_auc_norm,
        'precision_anom': precision_anom, 'recall_anom': recall_anom,
        'pr_thresholds_anom': pr_thresholds_anom,
        'pr_auc_anom': pr_auc_anom}
    return result_dict


def _transformations_experiment(dataset_load_fn, dataset_name,
    single_class_ind):
    """

    Args:
        dataset_load_fn:
        dataset_name:
        single_class_ind: inlier label value

    Returns:

    """
    (x_train, y_train), (x_test, y_test) = dataset_load_fn()

    if dataset_name in ['cats-vs-dogs']:
        transformer = Transformer(16, 16)
        n, k = (16, 8)
    else:
        transformer = Transformer(8, 8)
        n, k = (10, 4)
    mdl = create_wide_residual_network(x_train.shape[1:],
                                       transformer.n_transforms, n, k)
    mdl.compile('adam',
                'categorical_crossentropy',
                ['acc'])

    x_train_task = x_train[y_train.flatten() == single_class_ind]
    transformations_inds = np.tile(np.arange(transformer.n_transforms),
                                   len(x_train_task))
    x_train_task_transformed = transformer.transform_batch(
        np.repeat(x_train_task, transformer.n_transforms, axis=0),
        transformations_inds)
    batch_size = 128

    mdl.fit(x=x_train_task_transformed, y=to_categorical(transformations_inds),
            batch_size=batch_size,
            epochs=int(np.ceil(200 / transformer.n_transforms))
            )

    #################################################################################################
    # simplified normality score
    #################################################################################################
    # preds = np.zeros((len(x_test), transformer.n_transforms))
    # for t in range(transformer.n_transforms):
    #     preds[:, t] = mdl.predict(transformer.transform_batch(x_test, [t] * len(x_test)),
    #                               batch_size=batch_size)[:, t]
    #
    # labels = y_test.flatten() == single_class_ind
    # scores = preds.mean(axis=-1)
    #################################################################################################

    def calc_approx_alpha_sum(observations):
        N = len(observations)
        f = np.mean(observations, axis=0)

        return (N * (len(f) - 1) * (-psi(1))) / (
            N * np.sum(f * np.log(f)) - np.sum(
            f * np.sum(np.log(observations), axis=0)))

    def inv_psi(y, iters=5):
        # initial estimate
        cond = y >= -2.22
        x = cond * (np.exp(y) + 0.5) + (1 - cond) * -1 / (y - psi(1))

        for _ in range(iters):
            x = x - (psi(x) - y) / polygamma(1, x)
        return x

    def fixed_point_dirichlet_mle(alpha_init, log_p_hat, max_iter=1000):
        alpha_new = alpha_old = alpha_init
        for _ in range(max_iter):
            alpha_new = inv_psi(psi(np.sum(alpha_old)) + log_p_hat)
            if np.sqrt(np.sum((alpha_old - alpha_new) ** 2)) < 1e-9:
                break
            alpha_old = alpha_new
        return alpha_new

    def dirichlet_normality_score(alpha, p):
        return np.sum((alpha - 1) * np.log(p), axis=-1)

    scores = np.zeros((len(x_test),))
    observed_data = x_train_task
    for t_ind in range(transformer.n_transforms):
        observed_dirichlet = mdl.predict(
            transformer.transform_batch(observed_data,
                                        [t_ind] * len(observed_data)),
            batch_size=1024)
        log_p_hat_train = np.log(observed_dirichlet).mean(axis=0)

        alpha_sum_approx = calc_approx_alpha_sum(observed_dirichlet)
        alpha_0 = observed_dirichlet.mean(axis=0) * alpha_sum_approx

        mle_alpha_t = fixed_point_dirichlet_mle(alpha_0, log_p_hat_train)

        x_test_p = mdl.predict(
            transformer.transform_batch(x_test, [t_ind] * len(x_test)),
            batch_size=1024)
        scores += dirichlet_normality_score(mle_alpha_t, x_test_p)

    scores /= transformer.n_transforms
    labels = y_test.flatten() == single_class_ind

    # res_file_name = '{}_transformations_{}_{}.npz'.format(dataset_name,
    #                                                       get_class_name_from_index(
    #                                                           single_class_ind,
    #                                                           dataset_name),
    #                                                       datetime.now().strftime(
    #                                                           '%Y-%m-%d-%H%M'))
    # res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    results_dict = get_roc_pr_curve_data(scores, labels)

    # mdl_weights_name = '{}_transformations_{}_{}_weights.h5'.format(dataset_name,
    #                                                        get_class_name_from_index(single_class_ind, dataset_name),
    #                                                        datetime.now().strftime('%Y-%m-%d-%H%M'))
    # mdl_weights_path = os.path.join(RESULTS_DIR, dataset_name, mdl_weights_name)
    # mdl.save_weights(mdl_weights_path)
    return results_dict


def run_experiments(load_dataset_fn, dataset_name, n_runs):
    all_results_dicts = []
    # Transformations
    for _ in range(n_runs):
        all_results_dicts.append(
            _transformations_experiment(load_dataset_fn, dataset_name, 1))
    return all_results_dicts


def load_hits():
    hits_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '..', 'datasets', 'thesis_data', 'hits',
            'hits_small_4c_tuples.pkl'),
    }
    outlier_loader = HiTSOutlierLoaderv2(hits_params, 'small_hits')
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
    return (x_train, y_train), (x_test, y_test)


def load_ztf():
    ztf_params = {
        loader_keys.DATA_PATH: os.path.join(
            PROJECT_PATH, '..', 'datasets', 'thesis_data', 'ztfv7',
            'preprocessed_21', 'ztf_small_dict.pkl'),
    }
    outlier_loader = ZTFOutlierLoaderv2(ztf_params, 'small_ztfv7')
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = outlier_loader.get_outlier_detection_datasets()
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    N_RUNS = 1
    experiments_list = [
        (load_ztf, 'small_ztfv7'),
        (load_hits, 'small_hits')
    ]
    results = {}
    for data_load_fn, dataset_name in experiments_list:
        results[dataset_name] = run_experiments(
            data_load_fn, dataset_name, N_RUNS)

    for dataset_name in results.keys():
        print('\n',dataset_name)
        print('roc_auc: ', results[dataset_name][0]['roc_auc'])
        print('pr_auc_norm: ', results[dataset_name][0]['pr_auc_norm'])

    """
     small_ztfv7
    roc_auc:  0.8528262222222224
    pr_auc_norm:  0.8357925734356346
    
     small_hits
    roc_auc:  0.9923320000000001
    pr_auc_norm:  0.9906374967222835

    ztf
    Epoch 1/3
    504000/504000 [==============================] - 46s 91us/sample - loss: 1.5149 - acc: 0.5216
    Epoch 2/3
    504000/504000 [==============================] - 43s 86us/sample - loss: 0.9849 - acc: 0.7345
    Epoch 3/3
    504000/504000 [==============================] - 43s 86us/sample - loss: 0.8135 - acc: 0.8149
        
    hits
    Epoch 1/3
    504000/504000 [==============================] - 45s 89us/sample - loss: 2.3015 - acc: 0.1244
    Epoch 2/3
    504000/504000 [==============================] - 44s 87us/sample - loss: 2.1530 - acc: 0.1250
    Epoch 3/3
    504000/504000 [==============================] - 44s 87us/sample - loss: 2.1429 - acc: 0.1252
    """
