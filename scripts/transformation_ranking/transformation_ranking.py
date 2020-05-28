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

from modules.geometric_transform.transformer_no_compositions import \
  NoCompositionTransformer
from modules.data_loaders.artificial_dataset_factory import CirclesFactory

# class ResultDictGeneratorRawFID(object):
#
#   def __init__(self, data_loader: HiTSOutlierLoader):


if __name__ == "__main__":
  cicles_factory = CirclesFactory()
  samples = cicles_factory.get_final_dataset(n_images=10)

  from parameters import loader_keys, general_keys


