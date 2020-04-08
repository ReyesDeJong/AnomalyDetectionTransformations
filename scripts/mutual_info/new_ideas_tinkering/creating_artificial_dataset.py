"""
Creating an artificial dataset of circles with random sizes and random gaussian
 blurring
"""

import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
import matplotlib.pyplot as plt
from modules.geometric_transform.transformations_tf import makeGaussian, \
  cnn2d_depthwise_tf, check_shape_kernel


class CirclesFactory(object):
  def __init__(self, image_size=21, radius_range=(1, 4),
      sigma_gauss_range=(0, 3), gauss_kernel_size=7, random_seed=42):
    self.image_size = image_size
    self.radius_range = radius_range
    self.sigma_gauss_range = sigma_gauss_range
    self.gauss_kernel_size = gauss_kernel_size
    self.random_seed = random_seed
    self.sigma_noise = 0.1

  # TODO: maybe put outside;
  #  on utils, not necesarry as a method
  def create_circular_mask(self, h, w, center=None, radius=None):
    # use the middle of the image
    if center is None:
      center = [int(w / 2), int(h / 2)]
    # use the smallest distance between the center and image walls
    if radius is None:
      radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

  def get_circular_masks(self, number_of_masks_to_create):
    circular_masks_list = []
    all_random_radiuses = np.random.RandomState(self.random_seed).randint(
        self.radius_range[0], self.radius_range[1] + 1,
        size=number_of_masks_to_create)
    for i in range(number_of_masks_to_create):
      radius = all_random_radiuses[i]
      circle = self.create_circular_mask(self.image_size, self.image_size,
                                         radius=radius)
      circular_masks_list.append(circle)

    return np.array(circular_masks_list)

  # TODO: maybe put outside; on utils, not necesarry as a method
  def plot_n_images(self, image_array, save_path=None, plot_show=False,
      n_to_plot=25, fig_size=20, channel_to_plot=0, title=None, seed=42):
    if len(image_array.shape) == 4:
      image_array = image_array[..., channel_to_plot]

    n_imags_available = len(image_array)
    if n_imags_available < n_to_plot:
      n_to_plot = n_imags_available

    random_img_idxs = np.random.RandomState(seed).choice(
      range(n_imags_available), n_to_plot, replace=False)
    imgs_to_plot = image_array[random_img_idxs, ...]
    sqrt_n = int(np.ceil(np.sqrt(n_to_plot)))
    fig, axs = plt.subplots(
        sqrt_n, sqrt_n, figsize=(fig_size, fig_size),
        gridspec_kw={'wspace': 0.001, 'hspace': 0.002}, constrained_layout=True)
    if title:
      fig.suptitle(title, fontsize=40)  # , color='white')

    axs = axs.flatten()
    for img_index in range(n_to_plot):
      axs[img_index].imshow(imgs_to_plot[img_index])

    for ax in axs:
      ax.axis('off')

    # fig.tight_layout()
    if save_path:
      fig.savefig(os.path.join(save_path))

    if plot_show:
      plt.show()
    plt.close()

  def add_noise_to_back_ground(self, images, sigma_noise):
    masks = images == 0
    noise = np.random.normal(0, sigma_noise, masks.shape)
    masked_noise = masks * noise
    return images + masked_noise

  def add_noise(self, images, sigma_noise):
    noise = np.random.normal(0, sigma_noise, images.shape)
    return images + noise

  def gaussian_filter_images(self, images, kernel_size, gauss_sigma):
    kernel = makeGaussian(kernel_size, gauss_sigma).astype(np.float32)
    if len(images.shape) == 3:
      images = images[..., None]
    images = images.astype(np.float32)
    filtered_images = cnn2d_depthwise_tf(
        images, check_shape_kernel(kernel, images)).numpy()
    return filtered_images

  def apply_gaussian_filter_within_sigma_range(self, images):
    n_images_to_filter = len(images)
    all_random_sigmas = np.random.RandomState(self.random_seed).uniform(
        self.sigma_gauss_range[0], self.sigma_gauss_range[1],
        size=n_images_to_filter)
    filtered_images = []
    for i in range(n_images_to_filter):
      sigma = all_random_sigmas[i]
      image = images[i][None, ...]
      filtered_single_image = self.gaussian_filter_images(
          image, self.gauss_kernel_size, sigma)
      filtered_images.append(filtered_single_image[0])

    return np.array(filtered_images)

  def normilize_1_1(self, images):
    if len(images.shape) == 3:
      images = images[..., None]
    images -= np.nanmin(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
    images = images / np.nanmax(images, axis=(1, 2))[
                      :, np.newaxis, np.newaxis, :]
    images = 2 * images - 1
    return images

  def get_final_dataset(self, n_images):
    images = self.get_circular_masks(n_images)
    images = self.apply_gaussian_filter_within_sigma_range(images)
    images = self.normilize_1_1(images)
    images = self.add_noise(images, self.sigma_noise)
    images = self.normilize_1_1(images)
    images = np.float32(images)
    return images


if __name__ == '__main__':
  SHOW_PLOTS = True
  N_IMAGES = 500
  circle_factory = CirclesFactory()
  images = circle_factory.get_final_dataset(N_IMAGES)
  circle_factory.plot_n_images(images, plot_show=SHOW_PLOTS)
  print('')
