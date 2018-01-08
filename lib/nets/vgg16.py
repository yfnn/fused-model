# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg
import pdb

class vgg16(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)

  def _build_network(self, sess, is_training=True):
    with tf.variable_scope('vgg_16', 'vgg_16'):
      # select initializers
      if cfg.TRAIN.TRUNCATED:
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
      else:
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

      #thermal sub_net
      #with tf.device('/gpu:1'):
      net1 = slim.repeat(self._imageT, 2, slim.conv2d, 64, [3, 3],
                         trainable=False, scope='conv1T')
      #net1 = slim.conv2d(net1, 64, [1, 1],scope='NIN_T1')
      net1 = slim.max_pool2d(net1, [2, 2], padding='SAME', scope='pool1T')
      net1 = slim.repeat(net1, 2, slim.conv2d, 128, [3, 3],
                         trainable=False, scope='conv2T')
      #net1 = slim.conv2d(net1, 128, [1, 1],scope='NIN_T2')
      net1 = slim.max_pool2d(net1, [2, 2], padding='SAME', scope='pool2T')
      net1 = slim.repeat(net1, 3, slim.conv2d, 256, [3, 3],
                         trainable=is_training, scope='conv3T')
      #net1 = slim.conv2d(net1, 256, [1, 1],scope='NIN_T3')
      net1 = slim.max_pool2d(net1, [2, 2], padding='SAME', scope='pool3T')
      net1 = slim.repeat(net1, 3, slim.conv2d, 512, [3, 3],
                         trainable=is_training, scope='conv4T')
      #net1 = slim.max_pool2d(net1, [2, 2], padding='SAME', scope='pool4T')
      #net1 = slim.repeat(net1, 3, slim.conv2d, 512, [3, 3],
      #                   trainable=is_training, scope='conv5T')

      #RGB sub_net
      net2 = slim.repeat(self._imageRGB, 2, slim.conv2d, 64, [3, 3],
                         trainable=False, scope='conv1RGB')
      #net2 = slim.conv2d(net2, 64, [1, 1],scope='NIN_RGB1')
      net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool1RGB')
      net2 = slim.repeat(net2, 2, slim.conv2d, 128, [3, 3],
                         trainable=False, scope='conv2RGB')
      #net2 = slim.conv2d(net2, 128, [1, 1],scope='NIN_RGB2')
      net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool2RGB')
      net2 = slim.repeat(net2, 3, slim.conv2d, 256, [3, 3],
                         trainable=is_training, scope='conv3RGB')
      #net2 = slim.conv2d(net2, 256, [1, 1],scope='NIN_RGB3')
      net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool3RGB')
      net2 = slim.repeat(net2, 3, slim.conv2d, 512, [3, 3],
                         trainable=is_training, scope='conv4RGB')
      #net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool4RGB')
      #net2 = slim.repeat(net2, 3, slim.conv2d, 512, [3, 3],
      #                   trainable=is_training, scope='conv5RGB')

      pdb.set_trace()
      net = tf.concat([net1,net2],3,name='concat')
      net = slim.conv2d(net, 512, [1, 1],scope='NIN1')
      net = slim.conv2d(net, 512, [1, 1],scope='NIN2')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

      #net = tf.add(0.5*net1, 0.5*net2)

      self._act_summaries.append(net)
      self._layers['head'] = net
      # build the anchors for the image
      self._anchor_component()
      # region proposal network
      #pdb.set_trace()
      rois = self._region_proposal(net, is_training, initializer)
      # region of interest pooling
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net, rois, "pool5")
      else:
        raise NotImplementedError

      #fully connected layer
      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
      if is_training:
        fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True, scope='dropout6')
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
      if is_training:
        fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True, scope='dropout7')

      #global average pooling
      #fc7 = slim.avg_pool2d(pool5, [7, 7],padding='VALID',scope='avg_pool')
      
      # region classification
      #pdb.set_trace()
      cls_prob, bbox_pred = self._region_classification(fc7,
                                                        is_training,
                                                        initializer,
                                                        initializer_bbox)

      self._score_summaries.update(self._predictions)

      return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic_T, var_keep_dic_RGB):
    #variables_to_restore = []
    T_variables_to_restore = {}
    RGB_variables_to_restore = {}

    #for v in variables:
    #  # exclude the conv weights that are fc weights in vgg16
    #  if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
    #    self._variables_to_fix[v.name] = v
    #    continue
    #  # exclude the first conv layer to swap RGB to BGR
    #  if v.name == 'vgg_16/conv1RGB/conv1RGB_1/weights:0':
    #    self._variables_to_fix[v.name.replace('RGB','')] = v
    #    continue
    #  v1 = v.name.split(':')[0]
    #  if v1.replace('T','') in var_keep_dic:
    #    print('Variables restored: %s' % v.name)
    #    T_variables_to_restore[v1.replace('T','')]=v
    #    continue
    #  if v1.replace('RGB','') in var_keep_dic:
    #    print('Variables restored: %s' % v.name)
    #    RGB_variables_to_restore[v1.replace('RGB','')]=v
    #    continue

    for v in variables:
        # exclude the conv weights that are fc weights in vgg16
        if v.name == 'vgg_16/fc6/weights:0' or v.name == 'vgg_16/fc7/weights:0':
            self._variables_to_fix[v.name] = v
            continue
        # exclude the first conv layer to swap RGB to BGR
        if not(v.name[0:11]==u'vgg_16/conv'):
            continue
        if v.name == 'vgg_16/conv1RGB/conv1RGB_1/weights:0':
            self._variables_to_fix[v.name.replace('RGB','')] = v
            continue
        v1 = v.name.split(':')[0]
        if v1.replace('T','') in var_keep_dic_T:
            print('Variables restored: %s' % v.name)
            T_variables_to_restore[v1.replace('T','')]=v
            continue
        if v1.replace('RGB','') in var_keep_dic_RGB:
            print('Variables restored: %s' % v.name)
            RGB_variables_to_restore[v1.replace('RGB','')]=v
            continue

    return T_variables_to_restore,RGB_variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({"vgg_16/fc6/weights": fc6_conv,
                                      "vgg_16/fc7/weights": fc7_conv,
                                      "vgg_16/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix['vgg_16/fc6/weights:0'], tf.reshape(fc6_conv,
                            self._variables_to_fix['vgg_16/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix['vgg_16/fc7/weights:0'], tf.reshape(fc7_conv,
                            self._variables_to_fix['vgg_16/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix['vgg_16/conv1/conv1_1/weights:0'],
                            tf.reverse(conv1_rgb, [2])))
