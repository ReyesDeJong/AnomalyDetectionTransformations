"""
Training a model with basic-non-composed transforms, to visualize it feature
 layer and
then calculate FID
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

from scripts.transformation_ranking import transformation_ranking_v2, transformation_ranking_simple_trf



if __name__ == "__main__":
  print('')
  transformation_ranking_v2.main()
  transformation_ranking_simple_trf.main()