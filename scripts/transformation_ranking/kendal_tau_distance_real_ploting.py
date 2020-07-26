import itertools
import os
import sys

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_PATH)

import numpy as np

def kendall_tau_distance(order_a, order_b):
    pairs = itertools.combinations(range(1, len(order_a)+1), 2)
    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    denominator = len(order_a) * (len(order_a)-1) / 2
    norm_distance = distance / denominator
    return norm_distance

import matplotlib.pyplot as plt

def plot_thing(ticks, kendall, fontsize=12, title='HiTS'):
  plt.plot(np.arange(len(kendall)), kendall)
  plt.ylabel('Kendall Tau Distance', fontsize=fontsize)
  plt.xlabel('Ranking to be compared against Ground Truth Ranking', fontsize=fontsize)
  plt.title(title, fontsize=fontsize)
  # locs, labels = plt.xticks()  # Get the current locations and labels.
  # plt.xticks(np.arange(0, 1, step=0.2))  # Set label locations.
  # >> > xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
  plt.xticks([0, 1, 2], ticks,rotation = 20)  # Set text labels and properties.
  plt.show()

def plot_thing_both(ticks, kendall1, kendall2, fontsize=12):
  plt.plot(np.arange(len(kendall1)), kendall1, label='HiTS')
  plt.plot(np.arange(len(kendall2)), kendall2, label='ZTF')
  plt.ylabel('Kendall Tau Distance', fontsize=fontsize)
  plt.xlabel('Ranking to be compared against Ground Truth Ranking', fontsize=fontsize)
  plt.legend()
  # locs, labels = plt.xticks()  # Get the current locations and labels.
  # plt.xticks(np.arange(0, 1, step=0.2))  # Set label locations.
  # >> > xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.
  plt.xticks([0, 1, 2, 3], ticks,rotation = 20)  # Set text labels and properties.
  plt.show()

if __name__ == "__main__":
  # hits = ['ZTF', 'MNIST', 'Cifar10']
  # ztf = ['HiTS', 'MNIST', 'Cifar10']
  hits_kendall = {'Pixel\nShuffle': 0.4666666666666667, 'Cifar10': 0.6444444444444445, 'MNIST': 0.7555555555555555, 'Other\nAstro': 0.4444444444444444}
  ztf_kendall = {'Pixel\nShuffle': 0.6444444444444445, 'MNIST': 0.4444444444444444, 'Cifar10': 0.35555555555555557, 'Other\nAstro': 0.26666666666666666}
  labels = list(ztf_kendall.keys())
  hits_kendall_values = [hits_kendall[name] for name in labels]
  ztf_kendall_values = [ztf_kendall[name] for name in labels]


  # ztf = ['Other\nAstro', 'MNIST', 'Cifar10']
  # plot_thing(hits, hits_kendall)
  plot_thing_both(labels,hits_kendall_values, ztf_kendall_values)

