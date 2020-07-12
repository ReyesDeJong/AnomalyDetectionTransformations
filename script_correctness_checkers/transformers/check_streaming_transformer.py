import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname('__file__'), '..', '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
import tensorflow as tf
from modules.geometric_transform.streaming_transformers.transformer_ranking \
    import RankingTransformer
from tqdm import tqdm

def check_visualize_transforms():
    import imageio
    import os
    import matplotlib.pyplot as plt

    transformer = RankingTransformer()
    transformations_inds = np.arange(transformer.n_transforms)

    im_path = os.path.join(PROJECT_PATH, 'extra_files', 'dragon.png')
    im = imageio.imread(im_path)
    im = np.array([im[:150, :150, :3]]*transformer.n_transforms)
    im = im / np.max(im)
    im = 2 * im - 1
    print(im.shape)
    plt.imshow(im[0])
    plt.title('orig')
    plt.axis('off')
    plt.show()



    transformed_batch = \
        transformer.transform_batch_given_indexes(
        tf.convert_to_tensor(im, dtype=tf.float32),
        tf.convert_to_tensor(transformations_inds, dtype=tf.int32))

    transformed_batch = transformed_batch.numpy()
    print(transformed_batch)

    for i in range(transformer.n_transforms):
        transform_indx = i
        plt.imshow(transformed_batch[transform_indx])
        plt.title(str(transformer.transformation_tuples[i]))
        plt.axis('off')
        plt.show()

def check_memory_usage_when_changing_transforms(iterations=int(1e10)):
    transformer = RankingTransformer()
    trf_tuples = transformer.transformation_tuples
    for _ in tqdm(range(iterations)):
        trf_idxs = np.arange(len(trf_tuples))
        n_transforms_to_use = np.random.randint(low=1, high=len(trf_tuples))
        np.random.shuffle(trf_idxs)
        trf_idxs_to_select = trf_idxs[:n_transforms_to_use]
        trf_array = np.array(trf_tuples)[trf_idxs_to_select]
        trf_tuples_to_use = tuple(trf_array)
        transformer.set_transformations_to_perform(trf_tuples_to_use)

if __name__ == "__main__":
    check_visualize_transforms()
    # check_memory_usage_when_changing_transforms()