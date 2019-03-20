#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import cv2
import numpy as np
from glob import glob

import tensorflow as tf
from keras import backend as K

from model.pspnet import get_pspnet
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pspnet101_voc2012',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('-i', '--input_path', type=str, default='data/example_images/ade20k.jpg',
                        help='Path the input image')
    parser.add_argument('-g', '--glob_path', type=str, default=None,
                        help='Glob path for multiple images')
    parser.add_argument('-o', '--output_dir', type=str, default='output/',
                        help='Path to output')
    parser.add_argument('--id', default="0")
    parser.add_argument('-f', '--flip', type=bool, default=True,
                        help="Whether the network should predict on both image and flipped image.")

    parser.add_argument('-l', '--im_list', type=str, default=None)
    parser.add_argument('-d', '--im_dir', type=str, default=None)
    args = parser.parse_args()

    # Handle input and output args
    # images = glob(args.glob_path) if args.glob_path else [args.input_path,]
    # images.sort()

    im_list = []
    with open(args.im_list) as f:
        im_list = f.read().splitlines()

    # Directories
    cm_dir = os.path.join(args.output_dir, "cm/")
    pm_dir = os.path.join(args.output_dir, "pm/")
    vis_dir = os.path.join(args.output_dir, "vis/")

    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
        print(args)
        pspnet = get_pspnet(args.model, args.weights)
        for i, im_name in enumerate(im_list):
            print("Processing image {} / {} : {}".format(i+1,len(im_list), im_name))
            img_path = os.path.join(args.im_dir, im_name)
            cm_fn = join(cm_dir, im_name.replace('.jpg', '.png'))
            pm_fn = join(pm_dir, im_name.replace('.jpg', '.png'))
            vis_fn = join(vis_dir, im_name)
            if os.path.exists(vis_fn):
                continue

            img = cv2.imread(img_path)
            probs = pspnet.predict(img, args.flip)

            cm = np.argmax(probs, axis=2)
            pm = np.max(probs, axis=2)
            pm = np.array(pm*255, dtype='uint8')

            # color cm is [0.0-1.0] img is [0-255]
            color_cm = utils.add_color(cm)
            alpha_blended = 0.5 * color_cm * 255 + 0.

            if not os.path.exists(os.path.dirname(cm_fn)):
                os.makedirs(dirname(cm_fn))
            if not os.path.exists(os.path.dirname(pm_fn)):
                os.makedirs(dirname(pm_fn))
            if not os.path.exists(os.path.dirname(vis_fn)):
                os.makedirs(dirname(vis_fn))

            cv2.imwrite(cm_fn, cm)
            cv2.imwrite(pm_fn, pm)
            cv2.imwrite(vis_fn, alpha_blended)
