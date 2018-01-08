#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import pdb

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__',
           'person','people','cyclist','person?')


#NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
#DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
DATASETS= {'kaist': ('kaist_train',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_42570.ckpt',),'res101': ('res101_faster_rcnn_iter_4500.ckpt',)}
NUM_CLASSES = 5
pic_dir = '/home/yangfan/fused-model/tf-faster-rcnn/KAISTdevkit/data/images/set05/V000/'
an_dir = '/home/yangfan/fused-model/tf-faster-rcnn/KAISTdevkit/data/annotations/set05/V000/'
def vis_detections(im, class_name, dets, image_name, fig, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} {} detections with '
                  'p({} | box) >= {:.3f}').format(image_name, class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(sess, net, image_name_T, image_name_RGB):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    imRGB_file = os.path.join(cfg.DATA_DIR, 'demo', image_name_RGB)
    imRGB = cv2.imread(imRGB_file)
    imT_file = os.path.join(cfg.DATA_DIR, 'demo', image_name_T)
    imT = cv2.imread(imT_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, imT, imRGB)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.005
    NMS_THRESH = 0.3
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(imT, aspect='equal')
    #obj_num = len(an_lines) - 1
    #tags = [0]*obj_num

    for cls_ind, cls in enumerate(CLASSES[1:]):
        #pdb.set_trace()
        if cls == 'people' or cls == 'person?':
          continue
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        vis_detections(imRGB, cls, dets, image_name_RGB, fig, ax, thresh=CONF_THRESH)
        #if len(inds) == 0:
        #  continue

        #for index, obj in enumerate(an_lines):
        #    if(index == 0):
        #        continue
        #    if(tags[index-1] == 1):
        #        continue
        #    obj_splits = obj.split(' ')
        #    if obj_splits[0]=='people' or obj_splits[0]=='person?':
        #        continue
        #    for i in inds:
        #        bb = dets[i, :4]
        #        BBGT = obj_splits[1:5]
        #        ixmin = max(int(BBGT[0]), bb[0])
        #        iymin = max(int(BBGT[1]), bb[1])
        #        ixmax = min(int(BBGT[2])+int(BBGT[0])-1, bb[2])
        #        iymax = min(int(BBGT[3])+int(BBGT[1])-1, bb[3])
        #        iw = max(ixmax - ixmin + 1., 0.)
        #        ih = max(iymax - iymin + 1., 0.)
        #        inters = iw * ih

        #        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
        #                int(BBGT[2]) * int(BBGT[3]) - inters)
        #        overlap = inters / uni
        #        if(overlap >= 0.5):
        #            tags[index-1] = 1
        #            i = len(inds)
    #if(sum(tags) == obj_num):
        #print('################################')
    #    f.write(an_dir[-11:] + an_name[0:6]+'\n')
        #vis_detections(imT, cls, dets, fig, ax, image_name_T, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    #pdb.set_trace()
    with tf.device('/gpu:0'):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposalsi
        args = parse_args()

        # model path
        demonet = args.demo_net
        dataset = args.dataset
        tfmodel = os.path.join('/home/yangfan/fused-model/tf-faster-rcnn/output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
    #tfconfig = tf.ConfigProto(device_count={'GPU':0})
    tfconfig.gpu_options.allow_growth=True


    # init session
    sess = tf.Session(config=tfconfig)
    #pdb.set_trace()
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", NUM_CLASSES,
                          tag='default', anchor_scales=[4, 8, 16], anchor_ratios=[1,2])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    #pdb.set_trace()

    im_names = os.listdir(cfg.DATA_DIR + '/demo')
    #im_names.sort(key=lambda x:int(x[1:6]))
    #an_names = os.listdir(an_dir)
    #an_names.sort(key=lambda x:int(x[1:6]))
    im_rgb = '1.jpg'
    #f_selected = open('trainC.txt','a')
    #for i in xrange(100):
    for im_name in im_names:
      #f = open(an_dir + 'I' + im_name[1:6] + '.txt','r')
      #an_name = 'I' + im_name[1:6] + '.txt'
      #s = f.readlines()
      #if not (len(s)==1):
        #pdb.set_trace()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        #demo(sess, net, im_name, im_rgb, an_name, s, f_selected)
        demo(sess, net, im_name, im_rgb)
        #pdb.set_trace()
      #f.close()
    #f_selected.close()
    
    plt.show()
