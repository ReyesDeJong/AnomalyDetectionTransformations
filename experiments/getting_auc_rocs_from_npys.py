"""
Table I of paper, ztf_SOTA is the only one OK
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import numpy as np

a = np.array([[np.array(0.86454256), np.array(0.86712156), np.array(0.87361644),
               np.array(0.86451356), np.array(0.849454), np.array(0.87514422),
               np.array(0.83457789), np.array(0.84698833), np.array(0.841061),
               np.array(0.85324744)],
              [np.array(0.98027525), np.array(0.9857625), np.array(0.983536),
               np.array(0.9825625), np.array(0.984627), np.array(0.98488175),
               np.array(0.98796475), np.array(0.985407), np.array(0.98183725),
               np.array(0.98583125)],
              [np.array(0.7865), np.array(0.74916667), np.array(0.79833333),
               np.array(0.7925), np.array(0.755), np.array(0.78333333),
               np.array(0.75016667), np.array(0.7415), np.array(0.76066667),
               np.array(0.76133333)],
              [np.array(0.967), np.array(0.96825), np.array(0.97175),
               np.array(0.963), np.array(0.96975), np.array(0.9675),
               np.array(0.97125), np.array(0.966), np.array(0.9675),
               np.array(0.96725)],
              [np.array(0.82016667), np.array(0.801), np.array(0.82083333),
               np.array(0.8145), np.array(0.78383333), np.array(0.81566667),
               np.array(0.77983333), np.array(0.79183333), np.array(0.79),
               np.array(0.79383333)],
              [np.array(0.96825), np.array(0.9715), np.array(0.97175),
               np.array(0.96525), np.array(0.97175), np.array(0.969),
               np.array(0.9715), np.array(0.96875), np.array(0.97025),
               np.array(0.9695)],
              [np.array(0.80182077), np.array(0.82508328), np.array(0.81243268),
               np.array(0.80663329), np.array(0.79807117), np.array(0.83270923),
               np.array(0.76687149), np.array(0.78764613), np.array(0.77125922),
               np.array(0.7920947)],
              [np.array(0.94156055), np.array(0.96270831), np.array(0.95397577),
               np.array(0.9535592), np.array(0.96393102), np.array(0.96073936),
               np.array(0.97815279), np.array(0.96600179), np.array(0.95471655),
               np.array(0.9631495)]])

print(np.mean(a, axis=1))

print(np.std(a, axis=1))

npy_path = os.path.join(
    PROJECT_PATH, 'results', 'PAPER_RESULTS', 'HITS_ZTF_72_SOTA_VAL',
    'resnet_SOTA_epochs',
    'Transformer_OD_Model', 'all_metric_files', 'small_ztf')
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir(npy_path) if isfile(join(npy_path, f))]
aucroc = []

for filename in onlyfiles:
  path = os.path.join(npy_path, filename)
  npy_dict = dict(np.load(path))
  aucroc.append(npy_dict['roc_auc'])
  #print(npy_dict['roc_auc'])

print(np.mean(aucroc))
print(np.std(aucroc))
print(aucroc)
