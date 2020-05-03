"""
Table I of paper, ztf_SOTA is the only one OK
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

singlle_npy_path = os.path.join(
    PROJECT_PATH, 'HITS_ZTF_SOTA_VAL', 'resnet_SOTA_epochs', 'Transformer_OD_Model', 'all_metric_files')
