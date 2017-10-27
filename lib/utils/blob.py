# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  max_shape = np.array([im.shape for im in ims]).max(axis=0)
  #num_images = len(ims)
  num_images = int(len(ims)/2)
  blobT = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  blobRGB = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                     dtype=np.float32)
  for i in range(num_images):
    imT = ims[i*2]
    imRGB = ims[i*2+1]
    blobT[i, 0:imT.shape[0], 0:imT.shape[1], :] = imT
    blobRGB[i, 0:imRGB.shape[0], 0:imRGB.shape[1], :] = imRGB

  return blobT, blobRGB


def prep_im_for_blob(im, pixel_means, target_size, max_size):
  """Mean subtract and scale an image for use in a blob."""
  im = im.astype(np.float32, copy=False)
  im -= pixel_means
  im_shape = im.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  im_scale = float(target_size) / float(im_size_min)
  # Prevent the biggest axis from being more than MAX_SIZE
  if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)
  im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                  interpolation=cv2.INTER_LINEAR)

  return im, im_scale
