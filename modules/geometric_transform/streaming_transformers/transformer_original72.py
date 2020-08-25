import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)
from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
import itertools


class Original72Transformer(AbstractTransformer):
    def __init__(self, translation_x=8, translation_y=8, rotations=4,
        flips=True, gauss=False, log=False, mixed=0, trivial=0,
        transform_batch_size=512, name='original_72_Transformer',
        verbose=False):
        super().__init__(
            translation_x, translation_y, rotations, flips, gauss, log, mixed,
            trivial, transform_batch_size, name, verbose)

    def _get_transformation_tuples_list(self):
        transformation_tuples = list(
            itertools.product((False, True),
                              (0, -self.translation_x, self.translation_x),
                              (0, -self.translation_y, self.translation_y),
                              range(self.rotations)))
        for i, tup in enumerate(transformation_tuples):
            transformation_tuples[i] = tuple(list(tup) + [0, 0, 0, 0])
        return transformation_tuples


def test_visualize_transforms(transformer: AbstractTransformer):
    import imageio
    import os
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np

    im_path = os.path.join(PROJECT_PATH, 'extra_files', 'dragon.png')

    im = imageio.imread(im_path)

    im = im[np.newaxis, :150, :150, :3]
    im = im / np.max(im)
    print(im.shape)
    plt.imshow(im[0])
    plt.show()

    print(transformer.n_transforms)
    transformations_inds = np.arange(transformer.n_transforms)

    transformed_batch = transformer.transform_batch(
        tf.convert_to_tensor(im, dtype=tf.float32),
        transformations_inds)

    print(transformed_batch.shape)

    for i in range(transformer.n_transforms):
        transform_indx = i
        plt.imshow(transformed_batch[transform_indx])
        plt.title(str(transformer.transformation_tuples[i]))
        print(str(transformer.transformation_tuples[i]))
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    transformer = Original72Transformer()
    print(transformer.n_transforms)
    test_visualize_transforms(transformer)
