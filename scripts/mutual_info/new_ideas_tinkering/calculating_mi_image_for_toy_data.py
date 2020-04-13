"""
Calculating MI image over synth dataset, rot and traslation
No patches normalization works better.
Numpy on eigh values for MI estimations dont crush but gives slightly diff
results than with TF eigh
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
import matplotlib.pyplot as plt
from scripts.mutual_info.new_ideas_tinkering. \
  artificial_dataset_factory import CirclesFactory
from scripts.mutual_info.new_ideas_tinkering.mi_image_calculator import \
  InformationEstimatorByBatch, MIImageCalculator


def plot_mi_images(images_list, save_path=None, plot_show=False,
    single_img_fig_size=3, titles_list=None):
  n_images = len(images_list)
  fig, axs = plt.subplots(
      1, n_images,
      figsize=(single_img_fig_size * n_images, single_img_fig_size),
      gridspec_kw={'wspace': 0.001, 'hspace': 0.002}, constrained_layout=True)
  # if title:
  #   fig.suptitle(title, fontsize=40)  # , color='white')
  vmin = np.min(images_list)
  vmax = np.max(images_list)
  axs = axs.flatten()
  for img_index in range(n_images):
    img = axs[img_index].imshow(images_list[img_index], vmax=vmax, vmin=vmin)
    axs[img_index].set_title(titles_list[img_index])

  fig.colorbar(img)
  for ax in axs:
    ax.axis('off')
  # fig.tight_layout()
  if save_path:
    fig.savefig(os.path.join(save_path))
  if plot_show:
    plt.show()
  plt.close()


def mi_images_exp(images_without_noise, show_images, show_mi_images,
    transformation_shift, mi_image_calculator, normalize_patches,
    circle_factory):
  images_wn_translated = np.roll(images_without_noise,
                                 shift=transformation_shift, axis=2)
  images_wn_rotated = np.rot90(images_without_noise, axes=(1, 2))

  circle_factory.plot_n_images(images_without_noise, plot_show=show_images,
                               title='X')
  circle_factory.plot_n_images(
      images_wn_rotated, plot_show=show_images, title=r'$T_{rot90}(X)$')
  circle_factory.plot_n_images(
      images_wn_translated, plot_show=show_images,
      title=r'$T_{trans%ipx}(X)$' % transformation_shift)

  auto_mi_image = mi_image_calculator.mi_images_mean(
      images_without_noise, images_without_noise,
      normalize_patches=normalize_patches)
  rot_mi_image = mi_image_calculator.mi_images_mean(
      images_without_noise, images_wn_rotated,
      normalize_patches=normalize_patches)
  trans_mi_image = mi_image_calculator.mi_images_mean(
      images_without_noise, images_wn_translated,
      normalize_patches=normalize_patches)
  mi_images_list = [auto_mi_image, rot_mi_image, trans_mi_image]
  titles_list = [r'$I(X;X)$',
                 r'$I(X;T_{rot90}(X))$',
                 r'I(X;$T_{trans%ipx}(X))$' % transformation_shift]
  plot_mi_images(mi_images_list, plot_show=show_mi_images,
                 titles_list=titles_list)
  plot_mi_images(mi_images_list[1:], plot_show=show_mi_images,
                 titles_list=titles_list[1:])


if __name__ == '__main__':
  SHOW_IMAGES = False
  SHOW_MI_IMAGES = True
  BATCH_SIZE = 512
  N_IMAGES = 7000# BATCH_SIZE * 4
  WINDOW_SIZE = 3
  SIGMA_ZERO = 2.0
  TRANSFORMATION_SHIFT = 6
  NORMALIZE_PATCHES = True

  circle_factory = CirclesFactory()
  mi_estimator = InformationEstimatorByBatch(SIGMA_ZERO, BATCH_SIZE)
  mi_image_calculator = MIImageCalculator(information_estimator=mi_estimator,
                                          window_size=WINDOW_SIZE)

  # # EXP: images without noise
  images_without_noise = circle_factory.get_final_dataset_no_noise(N_IMAGES)
  # mi_images_exp(
  #     images_without_noise, SHOW_IMAGES, SHOW_MI_IMAGES, TRANSFORMATION_SHIFT,
  #     mi_image_calculator, NORMALIZE_PATCHES, circle_factory)

  # EXP: images without noise NO Norm_patches
  NORMALIZE_PATCHES = False
  # SHOW_IMAGES= False
  mi_images_exp(
      images_without_noise, SHOW_IMAGES, SHOW_MI_IMAGES, TRANSFORMATION_SHIFT,
      mi_image_calculator, NORMALIZE_PATCHES, circle_factory)
  # # EXP: images with noise
  # NORMALIZE_PATCHES = True
  # SHOW_IMAGES = False#True
  images_with_noise = circle_factory.get_final_dataset(N_IMAGES)
  # mi_images_exp(
  #     images_with_noise, SHOW_IMAGES, SHOW_MI_IMAGES, TRANSFORMATION_SHIFT,
  #     mi_image_calculator, NORMALIZE_PATCHES, circle_factory)

  # EXP: images with noise NO Norm_patches
  NORMALIZE_PATCHES = False
  # SHOW_IMAGES= False
  mi_images_exp(
      images_with_noise, SHOW_IMAGES, SHOW_MI_IMAGES, TRANSFORMATION_SHIFT,
      mi_image_calculator, NORMALIZE_PATCHES, circle_factory)
