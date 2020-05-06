# !/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import sys

"""
Must be run two times : 1 for download and 2 for extraction.
"""

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_PATH)



import os
import glob
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from modules.transform_selection.fid_modules import fid
from imageio import imread
import tensorflow as tf

# Paths
image_path = os.path.join(
    PROJECT_PATH, '..',
    'fid_extra_files/valid_64x64/valid_64x64_jpg')  # set path to some generated images
stats_path = os.path.join(
    PROJECT_PATH, '..',
    'fid_extra_files/fid_stats_imagenet_valid.npz') # training set statistics

stats_cifar_path = os.path.join(
    PROJECT_PATH, '..',
    'fid_extra_files/fid_stats_cifar10_train.npz')

inception_path = os.path.join(
    PROJECT_PATH, '..',
    'fid_extra_files/inception-2015-12-05')
inception_path = fid.check_or_download_inception(
  inception_path)  # download inception network

# loads all images into memory (this might require a lot of RAM!)
image_list = glob.glob(os.path.join(image_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
images = np.random.normal(0, 1, images.shape)

# load precalculated training set statistics
f = np.load(stats_path)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
print('real_data (npz) mu:%s sigma:%s' % (str(mu_real), str(sigma_real)))
f.close()

fid.create_inception_graph(
  inception_path)  # load the graph into the current TF graph
with tf.compat.v1.Session() as sess:
  sess.run(tf.compat.v1.global_variables_initializer())
  mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess,
                                                          batch_size=100)
  mu_real, sigma_real = fid.calculate_activation_statistics(images, sess,
                                                          batch_size=100)
  # f = np.load(stats_cifar_path)
  # mu_gen, sigma_gen = f['mu'][:], f['sigma'][:]
print('images (jpg) mu:%s sigma:%s' % (str(mu_gen), str(sigma_gen)))
fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real,
                                           sigma_real)
print("FID: %s" % fid_value)
