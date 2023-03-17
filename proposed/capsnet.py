# encoding = utf8
import numpy as np
import tensorflow as tf
from config import Config
#
cfg = Config()


class CapsNet(object):
    ''' 胶囊层
    参数：
        input：一个4维张量。
        num_units：整数，胶囊的输出向量的长度。
        with_routing：布尔值，该胶囊路由经过低层胶囊。
        num_outputs：该层中的胶囊数目。
    返回：
        一个4维张量。
    '''
    def __init__(self, num_units, with_routing=True):
        self.num_units = num_units
        self.with_routing = with_routing

    def __call__(self, input, num_outputs, kernel_size=None, stride=None):
        self.num_outputs = num_outputs
        self.kernel_size = kernel_size
        self.stride = stride

        if not self.with_routing:
            # 主胶囊（PrimaryCaps）层
            capsules = []
            for i in range(self.num_units):
                with tf.variable_scope('ConvUnit_' + str(i)):
                    caps_i = tf.contrib.layers.conv2d(input,
                                                      self.num_outputs,
                                                      self.kernel_size,
                                                      self.stride,
                                                      padding="VALID")
                    caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
                    capsules.append(caps_i)

            capsules = tf.concat(capsules, axis=2)
            capsules = self.squash(capsules)

        else:
            # 数字胶囊（DigitCaps）层
            self.input = tf.reshape(input, shape=(cfg.batch_size, 32, 8, 1))
            b_IJ = tf.zeros(shape=[1, 128, 32, 1], dtype=np.float32)
            capsules = []
            for j in range(self.num_outputs):
                with tf.variable_scope('caps_' + str(j)):
                    caps_j, b_IJ = self.capsule(input, b_IJ, j)
                    capsules.append(caps_j)

            capsules = tf.concat(capsules, axis=1)
            print('capsules:', capsules)
            assert capsules.get_shape() == [cfg.batch_size, 8, 21, 1]

        return capsules

    def capsule(self, input, b_IJ, idx_j):
        input = input[:, : 128, :, :]
        with tf.variable_scope('routing'):
            w_initializer = np.random.normal(size=[1, 128, 8, 21], scale=0.01)
            W_Ij = tf.Variable(w_initializer, dtype=tf.float32)
            W_Ij = tf.tile(W_Ij, [cfg.batch_size, 1, 1, 1])
            u_hat = tf.matmul(W_Ij, input, transpose_a=True)
            assert u_hat.get_shape() == [cfg.batch_size, 128, 21, 1]

            shape = b_IJ.get_shape().as_list()
            size_splits = [idx_j, 1, shape[2] - idx_j - 1]
            for r_iter in range(cfg.iter_routing):
                c_IJ = tf.nn.softmax(b_IJ, dim=2)
                print('c_IJ:', c_IJ)
                assert c_IJ.get_shape() == [1, 128, 32, 1]
                b_Il, b_Ij, b_Ir = tf.split(b_IJ, size_splits, axis=2)
                c_Il, c_Ij, b_Ir = tf.split(c_IJ, size_splits, axis=2)
                assert c_Ij.get_shape() == [1, 128, 1, 1]
                s_j = tf.reduce_sum(tf.multiply(c_Ij, u_hat), axis=1, keep_dims=True)
                assert s_j.get_shape() == [cfg.batch_size, 1, 21, 1]

                v_j = self.squash(s_j)
                assert s_j.get_shape() == [cfg.batch_size, 1, 21, 1]

                v_j_tiled = tf.tile(v_j, [1, 128, 1, 1])
                u_produce_v = tf.matmul(u_hat, v_j_tiled, transpose_a=True)
                assert u_produce_v.get_shape() == [cfg.batch_size, 128, 1, 1]
                b_Ij += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ = tf.concat([b_Il, b_Ij, b_Ir], axis=2)

            return v_j, b_IJ

    def squash(self, vector):
        '''压缩函数
        参数：
            vector：一个4维张量 [batch_size, num_caps, vec_len, 1],
        返回：
            一个和vector形状相同的4维张量，
            但第3维和第4维经过压缩
        '''
        vec_abs = tf.sqrt(tf.reduce_sum(tf.square(vector)))  # 一个标量
        scalar_factor = tf.square(vec_abs) / (1 + tf.square(vec_abs))
        vec_squashed = scalar_factor * tf.divide(vector, vec_abs)  # 对应元素相乘
        return vec_squashed