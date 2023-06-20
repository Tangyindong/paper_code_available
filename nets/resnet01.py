from keras.layers import ZeroPadding2D,Conv2D,MaxPooling2D,BatchNormalization,Activation,Add

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.initializers as KI
import keras.regularizers as KR
import keras.constraints as KC
from keras.engine import InputSpec
from keras.utils import conv_utils

import utils

def log(text, array=None):
    
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else "",
            array.dtype))
    print(text)

class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)
def compute_backbone_shapes(config, image_shape):    
    assert config.BACKBONE in ["resnet50", "resnet101", "mobilenet224v1"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])
def relu6(x):
    return K.relu(x, max_value=6)
class DepthwiseConv2D(KL.Conv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = KI.get(depthwise_initializer)
        self.depthwise_regularizer = KR.get(depthwise_regularizer)
        self.depthwise_constraint = KC.get(depthwise_constraint)
        self.bias_initializer = KI.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True
    
    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = KI.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = KR.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = KC.serialize(self.depthwise_constraint)
        return config

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), train_bn=True):
  
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = KL.Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNorm(axis=channel_axis, name='conv1_bn')(x, training=train_bn)
    return KL.Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), 
                          block_id=1, train_bn=True):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNorm(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x, training=train_bn)
    x = KL.Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = KL.Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNorm(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x, training=train_bn)
    return KL.Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def get_resnet(img_input,alpha=1.0, depth_multiplier=1, train_bn=True):
    # assert architecture in ["mobilenet224v1"]
    # Stage 1
    # 512 512 3->256 256 32
    x = _conv_block(img_input, 32, alpha, strides=(2, 2), train_bn=train_bn)
    C1 = x = _depthwise_conv_block(x, 64, alpha, 
                                   depth_multiplier, block_id=1, train_bn=train_bn)
    # 256 256 32->128 128 128
    # Stage 2
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, train_bn=train_bn)
    C2 = x = _depthwise_conv_block(x, 128, alpha, 
                                   depth_multiplier, block_id=3, train_bn=train_bn)
    # 128 128 128->64 64 256
    # Stage 3
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, train_bn=train_bn)
    C3 = x = _depthwise_conv_block(x, 256, alpha, 
                                   depth_multiplier, block_id=5, train_bn=train_bn)
    # 64 64 256 ->32 32 512
    # Stage 4
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6, train_bn=train_bn)
    x = _depthwise_conv_block(x, 512, alpha, 
                              depth_multiplier, block_id=7, train_bn=train_bn)
    x = _depthwise_conv_block(x, 512, alpha, 
                              depth_multiplier, block_id=8, train_bn=train_bn)
    x = _depthwise_conv_block(x, 512, alpha, 
                              depth_multiplier, block_id=9, train_bn=train_bn)
    x = _depthwise_conv_block(x, 512, alpha, 
                              depth_multiplier, block_id=10, train_bn=train_bn)
    C4 = x = _depthwise_conv_block(x, 512, alpha, 
                                   depth_multiplier, block_id=11, train_bn=train_bn)
    # 32 32 512->16 16 1024
    # Stage 5
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12, train_bn=train_bn)
    C5 = x = _depthwise_conv_block(x, 1024, alpha, 
                                   depth_multiplier, block_id=13, train_bn=train_bn)
    return [C1, C2, C3, C4, C5]
