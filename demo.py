#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import cv2
import numpy as np

import tensorflow as tf
from keras import backend as K

from model.pspnet import get_pspnet
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-f', '--flip', type=bool, default=True,
                        help="Whether the network should predict on both image and flipped image.")
    parser.add_argument('-i', '--image_path', type=str, default='data/example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-o', '--output_dir', type=str, default='./output/',
                        help='Path to output directory')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():

        pspnet = get_pspnet(args.model, args.weights)
        img = cv2.imread(args.image_path)

        # Predict
        probs = pspnet.predict(img, args.flip)
        cm = np.argmax(probs, axis=2)
        pm = np.max(probs, axis=2)
        pm = np.array(pm*255, dtype='uint8')

        # Visualize
        # color cm is [0.0-1.0] img is [0-255]
        color_cm = utils.add_color(cm)
        alpha_blended = 0.5 * color_cm * 255 + 0.5 * img

        # Save
        im_name = os.path.basename(args.image_path)
        cm_fn = os.path.join(args.output_dir, im_name.replace('.jpg', '_cm.png'))
        pm_fn = os.path.join(args.output_dir, im_name.replace('.jpg', '_pm.png'))
        vis_fn = os.path.join(args.output_dir, im_name.replace('.jpg', '_vis.png'))
        cv2.imwrite(cm_fn, cm)
        cv2.imwrite(pm_fn, pm)
        cv2.imwrite(vis_fn, alpha_blended)
