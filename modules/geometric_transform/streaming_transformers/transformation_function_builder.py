import tensorflow as tf
from modules.geometric_transform. \
    transformations_tf import cnn2d_depthwise_tf, \
    makeGaussian, makeLoG, check_shape_kernel, apply_affine_transform

class TransformationFunctionBuilder(object):
    def __init__(self, flip, tx, ty, k_90_rotate, gauss, log, mixed, trivial):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.k_90_rotate = k_90_rotate
        self.gauss = gauss
        self.log = log
        self.gauss_kernel = tf.constant(makeGaussian(5, 1), tf.float32)
        self.log_kernel = tf.constant(makeLoG(5, 0.5), tf.float32)
        self.mixed = mixed
        self.trivial = trivial

    def __call__(self, x):
        res_x = x
        if self.gauss:
            res_x = cnn2d_depthwise_tf(
                res_x, check_shape_kernel(self.gauss_kernel, res_x))
        if self.log:
            res_x = cnn2d_depthwise_tf(
                res_x, check_shape_kernel(self.log_kernel, res_x))
        if self.flip:
                res_x = tf.image.flip_left_right(res_x)
        if self.tx != 0 or self.ty != 0:
            res_x = apply_affine_transform(res_x, self.tx, self.ty)
        if self.k_90_rotate != 0:
                res_x = tf.image.rot90(res_x, k=self.k_90_rotate)
        if self.mixed:
            flatten_img_x = tf.reshape(x, [x.shape[0], x.shape[1] * x.shape[2],
                                           x.shape[3]])
            perm_x = tf.transpose(flatten_img_x, [1, 0, 2])
            shufled_x = tf.random.shuffle(perm_x)
            perm2_x = tf.transpose(shufled_x, [1, 0, 2])
            reshaped_x = tf.reshape(perm2_x, [x.shape[0], x.shape[1],
                                              x.shape[2], x.shape[3]])
            res_x = reshaped_x
        if self.trivial:
            res_x = x * 0 + tf.random.normal(x.shape)
        return res_x