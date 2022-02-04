""" We owe the following implementation of [Li et al.'s involution layer](https://arxiv.org/abs/2103.06255) 
to this GitHub repository : https://github.com/ariG23498/involution-tf. 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Involution(tf.keras.layers.Layer):
    def __init__(self, channel, group_number, kernel_size, stride, reduction_ratio):
        super().__init__()
        # The assert makes sure that the user knows about the
        # reduction size. We cannot have 0 filters in Conv2D.
        assert reduction_ratio <= channel, print("Reduction ration must be less than or equal to channel size")
        
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        self.o_weights = tf.keras.layers.AveragePooling2D(
            pool_size=self.stride,
            strides=self.stride,
            padding="same") if self.stride > 1 else tf.identity
        self.kernel_gen = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=self.channel//self.reduction_ratio,
                kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                filters=self.kernel_size*self.kernel_size*self.group_number,
                kernel_size=1)
        ])

    def call(self, x):
        _, H, W, C = x.shape
        H = H//self.stride
        W = W//self.stride
        # Extract input feature blocks
        unfolded_x = tf.image.extract_patches(
            images=x,
            sizes=[1,self.kernel_size,self.kernel_size,1],
            strides=[1,self.stride,self.stride,1],
            rates=[1,1,1,1],
            padding="SAME")                                                     # B, H, W, K*K*C
        unfolded_x = tf.keras.layers.Reshape(
            target_shape=(H,
                   W,
                   self.kernel_size*self.kernel_size,
                   C//self.group_number,
                   self.group_number)
            )(unfolded_x)                                                       # B, H, W, K*K, C//G, G

        # generate the kernel
        kernel_inp = self.o_weights(x)
        kernel = self.kernel_gen(kernel_inp)                                    # B, H, W, K*K*G
        kernel = tf.keras.layers.Reshape(
            target_shape=(H,
                   W,
                   self.kernel_size*self.kernel_size,
                   1,
                   self.group_number)
            )(kernel)                                                           # B, H, W, K*K, 1, G

        # Multiply-Add op
        out = tf.math.multiply(kernel, unfolded_x)                              # B, H, W, K*K, C//G, G
        out = tf.math.reduce_sum(out, axis=3)                                   # B, H, W, C//G, G
        out = tf.keras.layers.Reshape(
            target_shape=(H,
                W,
                C)
        )(out)                                                                  # B, H, W, C
        return out