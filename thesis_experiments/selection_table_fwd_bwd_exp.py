"""
Training a model with basic-non-composed transforms, to visualize it feature
 layer and
then calculate FID
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

from modules.geometric_transform import transformations_tf
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
from modules.geometric_transform.transformer_for_ranking import \
    RankingTransformer
from modules.data_loaders.ztf_small_outlier_loader import \
    ZTFSmallOutlierLoader
from modules.transform_selection.pipeline.trivial_selector import \
    TrivialTransformationSelector
from modules.transform_selection.pipeline.disc_matrix_selector import \
    DiscriminationMatrixTransformationSelector
from modules.transform_selection.pipeline.rank_fwd_selector import \
    RankingForwardTransformationSelector
from modules.transform_selection.pipeline.rank_bwd_selector import \
    RankingBackwardTransformationSelector
from modules.transform_selection.pipeline.pipeline_selection import \
    PipelineTransformationSelection
from modules.print_manager import PrintManager
from modules.utils import check_path

# TF logging control
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_dataset_loaders_list():
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

    return [hits_loader, ztf_loader]

def get_pipelines_list(
    verbose_pipeline, verbose_selectors, transform_from_scratch):
    pipeline_c1_c2b_c3fwd = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                TrivialTransformationSelector(),
                DiscriminationMatrixTransformationSelector(
                    from_scartch=False),
                RankingForwardTransformationSelector(
                    transformations_from_scratch=transform_from_scratch)
            ]
        )
    pipeline_c1_c2b_c3bwd = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                TrivialTransformationSelector(),
                DiscriminationMatrixTransformationSelector(
                    from_scartch=False),
                RankingBackwardTransformationSelector(
                    transformations_from_scratch=transform_from_scratch)
            ]
        )
    return [pipeline_c1_c2b_c3fwd, pipeline_c1_c2b_c3bwd]

def get_transformers_constructors():
    trf_ranking = RankingTransformer
    trf_72 = transformations_tf.Transformer
    trf_99 = transformations_tf.PlusKernelTransformer
    return [trf_ranking, trf_72, trf_99]


def main():
    VERBOSE_PIPELINE = True
    VERBOSE_SELECTORS = True
    TRANSFORM_FROM_SCRATCH = True

    results_folder_path = os.path.join(
        PROJECT_PATH, 'results', 'transformation_selection',
        'pipelines', 'logs')
    check_path(results_folder_path)

    for transformer_constructor in get_transformers_constructors():
        for pipeline in get_pipelines_list(
            VERBOSE_PIPELINE, VERBOSE_SELECTORS, TRANSFORM_FROM_SCRATCH):
            for dataset_loader in get_dataset_loaders_list():
                transformer = transformer_constructor()
                print_manager = PrintManager()
                pipeline_name = pipeline.get_pipeline_name(
                    transformer, dataset_loader, transformer.n_transforms)
                if 'Bwd' in pipeline_name and transformer.n_transforms > 36:
                    continue
                log_file_name = '%s.log' % pipeline_name
                log_file_path = os.path.join(results_folder_path, log_file_name)
                log_file = open(log_file_path, 'w')
                print_manager.file_printing(log_file)
                print(pipeline_name)
                print('Init N transforms %i\n%s' % (
                    transformer.n_transforms,
                    str(transformer.transformation_tuples)))
                (x_train, y_train), _, _ = dataset_loader.\
                    get_outlier_detection_datasets()
                transformer = pipeline.get_selected_transformer(
                    transformer, x_train, dataset_loader)
                print_manager.file_printing(log_file)
                print('Final N transforms %i\n%s' % (
                    transformer.n_transforms,
                    str(transformer.transformation_tuples)))
                print_manager.close()

if __name__ == "__main__":
    main()
