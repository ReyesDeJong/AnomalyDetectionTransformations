import numpy as np
import torch


# TODO see if its faster or better to us xH
def get_xH(transformer, matrix_evals):
  matrix_evals[matrix_evals == 0] += 1e-10
  matrix_evals[matrix_evals == 1] -= 1e-10
  matrix_evals_compĺement = 1 - matrix_evals

  matrix_vals_stack = np.stack([matrix_evals_compĺement, matrix_evals],
                               axis=-1)
  xH = torch.nn.NLLLoss(reduction='none')
  gt_matrix = np.stack(
      [np.eye(transformer.n_transforms)] * len(matrix_vals_stack))
  gt_torch = torch.LongTensor(gt_matrix)
  matrix_logSoftmax_torch = torch.FloatTensor(
      np.swapaxes(np.swapaxes(matrix_vals_stack, 1, -1), -1, -2)).log()
  loss_xH = xH(matrix_logSoftmax_torch, gt_torch)
  batch_xH = np.mean(loss_xH.numpy(), axis=(-1, -2))
  return batch_xH


def get_entropy(matrix_scores, epsilon=1e-10):
  norm_scores = matrix_scores / np.sum(matrix_scores, axis=(1, 2))[
    ..., np.newaxis, np.newaxis]
  norm_scores[norm_scores == 0] += epsilon
  log_scores = np.log(norm_scores)
  product = norm_scores * log_scores
  entropy = -np.sum(product, axis=(1, 2))
  return entropy


def get_xy_mutual_info(matrix_scores, epsilon=1e-10):
  matrix_scores[matrix_scores == 0] += epsilon
  norm_scores = matrix_scores / np.sum(matrix_scores, axis=(1, 2))[
    ..., np.newaxis, np.newaxis]
  I = 0
  for x in range(matrix_scores.shape[1]):
    for y in range(matrix_scores.shape[2]):
      I += norm_scores[:, x, y] * np.log(norm_scores[:, x, y] / (
            norm_scores[:, x, :].sum(axis=-1) * norm_scores[:, :, y].sum(
          axis=-1)))
  return I


if __name__ == '__main__':
  a = np.eye(5)
  b = np.eye(5)
  b[0, 0] = 0.8
  b[0, 2] = 0.2
  c = np.eye(5)
  c[2, 0] = 0.2
  c[2, 2] = 0.8
  c[3, 0] = 0.2
  c[3, 3] = 0.8
  mx = np.stack([a, b, c], axis=0)
  MI = get_xy_mutual_info(mx)
