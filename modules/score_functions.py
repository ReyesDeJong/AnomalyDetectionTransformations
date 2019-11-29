import torch
import numpy as np

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
