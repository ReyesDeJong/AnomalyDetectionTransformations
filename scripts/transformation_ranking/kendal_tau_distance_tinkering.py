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

if __name__ == "__main__":
  a = np.arange(100)+1
  b = a.copy()
  np.random.shuffle(b)
  print(a)
  print(b)
  # print(kendall_tau_distance(b.tolist(), a.tolist()))
  print(kendall_tau_distance([1,2,3], [1,3,2]))