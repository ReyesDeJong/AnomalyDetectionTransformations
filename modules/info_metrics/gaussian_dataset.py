import numpy as np
# From Tapia-Reyes work

def generate_dataset(dimension, corr_factor, n_samples):
    # Joint Distribution
    identity = np.identity(dimension)
    joint_covariance = np.concatenate(
        [np.concatenate([identity, corr_factor*identity], axis=1),
         np.concatenate([corr_factor*identity, identity], axis=1)], axis=0
    )
    joint_mean = np.zeros(2*dimension)
    # Theoretical Mutual Information
    mut_info = -(1/2) * np.log(np.linalg.det(joint_covariance))
    # Sampling dataset
    samples = np.random.multivariate_normal(joint_mean, joint_covariance, size=n_samples)
    samples_x = samples[:, :dimension]
    samples_z = samples[:, dimension:]
    return samples_x, samples_z, mut_info
