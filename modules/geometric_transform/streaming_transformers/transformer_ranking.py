from modules.geometric_transform.streaming_transformers. \
    abstract_streaming_transformer import AbstractTransformer
import numpy as np


class RankingTransformer(AbstractTransformer):
    def __init__(self, translation_x=8, translation_y=8, rotations=True,
        flips=True, gauss=True, log=True, mixed=1, trivial=1,
        transform_batch_size=512, name='Stream_Ranking_Transformer'):
        super().__init__(translation_x, translation_y, rotations, flips, gauss,
                         log, mixed, trivial, transform_batch_size, name)

    def _get_transformation_tuples_list(self):
        transformation_tuples = (
            (0, 0, 0, 0, 0, 0, 0, 0), (1 * self.flips, 0, 0, 0, 0, 0, 0, 0),
            (0, self.translation_x, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, self.rotations * 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 1 * self.gauss, 0, 0, 0),
            (0, 0, 0, 0, 0, 1 * self.log, 0, 0),
            (0, 0, 0, 0, 0, 0, self.mixed, 0),
            (0, 0, 0, 0, 0, 0, 0, self.trivial),
        )
        # if some of the parameters is st to zero, avoid transformation
        # redundance, because original would appear more than once
        if self.translation_y * self.translation_x * self.rotations * \
            self.flips * self.gauss * self.log * self.trivial * self.mixed == 0:
            transformation_tuples = tuple(
                np.unique(transformation_tuples, axis=0))
        return transformation_tuples
