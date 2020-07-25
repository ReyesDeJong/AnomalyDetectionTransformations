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

from models.transformer_od import TransformODModel
from modules.trainer import ODTrainer
from modules.geometric_transform.transformations_tf import AbstractTransformer
from modules.data_loaders.hits_outlier_loader import HiTSOutlierLoader
from parameters import loader_keys, general_keys
from modules.geometric_transform.transformer_for_ranking_with_comp import \
    RankingSingleCompositionTransformer
from modules.data_loaders.ztf_small_outlier_loader import \
    ZTFSmallOutlierLoader
from modules.transform_selection.pipeline.trivial_selector import \
    TrivialTransformationSelector
from modules.transform_selection.pipeline.fid_selector import \
    FIDTransformationSelector
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
from parameters import param_keys

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

    return [
        hits_loader, 
        #ztf_loader
    ]

def get_pipelines_list(
    verbose_pipeline, verbose_selectors, transform_from_scratch):
    pipeline_c1 = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                TrivialTransformationSelector(),
            ]
        )
    pipeline_c2a = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                DiscriminationMatrixTransformationSelector(
                    from_scartch=transform_from_scratch)
            ]
        )
    pipeline_c2b = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                FIDTransformationSelector(),
            ]
        )

    pipeline_c3fwd = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                RankingForwardTransformationSelector(
                    transformations_from_scratch=transform_from_scratch)
            ]
        )

    pipeline_c3bwd = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                RankingBackwardTransformationSelector(
                    transformations_from_scratch=transform_from_scratch)
            ]
        )

    pipeline_c1_c2a = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                TrivialTransformationSelector(),
                DiscriminationMatrixTransformationSelector(
                    from_scartch=transform_from_scratch)
            ]
        )
    pipeline_c1_c2b = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                TrivialTransformationSelector(),
                FIDTransformationSelector(),
            ]
        )
    pipeline_c1_c2a_c3fwd = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                TrivialTransformationSelector(),
                DiscriminationMatrixTransformationSelector(
                    from_scartch=transform_from_scratch),
                RankingForwardTransformationSelector(
                    transformations_from_scratch=transform_from_scratch)
            ]
        )

    pipeline_c1_c2b_c3fwd = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                TrivialTransformationSelector(),
                FIDTransformationSelector(),
                RankingForwardTransformationSelector(
                    transformations_from_scratch=transform_from_scratch)
            ]
        )

    pipeline_c1_c2a_c3bwd = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                TrivialTransformationSelector(),
                DiscriminationMatrixTransformationSelector(
                    from_scartch=transform_from_scratch),
                RankingBackwardTransformationSelector(
                    transformations_from_scratch=transform_from_scratch)
            ]
        )

    pipeline_c1_c2b_c3bwd = \
        PipelineTransformationSelection(
            verbose_pipeline=verbose_pipeline,
            verbose_selectors=verbose_selectors,
            selection_pipeline=[
                TrivialTransformationSelector(),
                FIDTransformationSelector(),
                RankingBackwardTransformationSelector(
                    transformations_from_scratch=transform_from_scratch)
            ]
        )
    return [
        pipeline_c1, pipeline_c2a, 
        pipeline_c2b, 
        pipeline_c1_c2a,
        pipeline_c1_c2b, pipeline_c1_c2a_c3fwd, pipeline_c1_c2b_c3fwd,
        pipeline_c1_c2a_c3bwd, pipeline_c1_c2b_c3bwd, pipeline_c3fwd,
        pipeline_c3bwd
    ]

def get_transformers_constructors():
    trf_ranking = RankingSingleCompositionTransformer
    return [trf_ranking]

def evaluate_pipeline_transformer(
    train_mesage_transformer, transformer: AbstractTransformer, data_loader:HiTSOutlierLoader):
    (x_train, y_train), (x_val, y_val), (
        x_test, y_test) = data_loader.get_outlier_detection_datasets()
    model_trainer = ODTrainer(
        {param_keys.EPOCHS: 10000})
    model_trainer.train_and_evaluate_model_n_times(
        TransformODModel, transformer, x_train, x_val, x_test, y_test,
        10, False)
    result_mean, result_var = model_trainer.get_metric_mean_and_std()
    print('\n[_RESULTS] %s_%s %i %.5f+/-%.5f' % (
        train_mesage_transformer, data_loader.name, transformer.n_transforms,
        result_mean, result_var))





def main():
    VERBOSE_PIPELINE = True
    VERBOSE_SELECTORS = True
    TRANSFORM_FROM_SCRATCH = True

    results_folder_path = os.path.join(
        PROJECT_PATH, 'results', 'transformation_selection',
        'pipelines', 'logs')
    check_path(results_folder_path)
    data_loader_counter=0
    for transformer_constructor in get_transformers_constructors():
        for pipeline in get_pipelines_list(
            VERBOSE_PIPELINE, VERBOSE_SELECTORS, TRANSFORM_FROM_SCRATCH):
            for dataset_loader in get_dataset_loaders_list():
                transformer = transformer_constructor()
                print_manager = PrintManager()
                pipeline_name = pipeline.get_pipeline_name(
                    transformer, dataset_loader, transformer.n_transforms)
                log_file_name = '%s.log' % pipeline_name
                log_file_path = os.path.join(results_folder_path, log_file_name)
                log_file = open(log_file_path, 'w')
                print_manager.file_printing(log_file)
                print(pipeline_name)
                print('Init N transforms %i\n%s' % (
                    transformer.n_transforms,
                    str(transformer.transformation_tuples)))
                if data_loader_counter<len(get_dataset_loaders_list()):
                    data_loader_counter+=1
                    evaluate_pipeline_transformer('INITIAL', transformer,
                                                  dataset_loader)
                (x_train, y_train), _, _ = dataset_loader.\
                    get_outlier_detection_datasets()
                transformer = pipeline.get_selected_transformer(
                    transformer, x_train, dataset_loader)
                print_manager.close()
                print_manager.file_printing(log_file)
                print('Final N transforms %i\n%s' % (
                    transformer.n_transforms,
                    str(transformer.transformation_tuples)))
                print(pipeline_name)
                evaluate_pipeline_transformer(pipeline_name, transformer,
                                              dataset_loader)
                print_manager.close()
                log_file.close()

if __name__ == "__main__":
    main()
