#!/usr/bin/env python
from __future__ import print_function
import os
from os.path import splitext, join, isfile, isdir, basename, exists, dirname
import argparse
import cv2
import numpy as np
from glob import glob
from scipy import ndimage

import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json, load_model
from keras.utils.generic_utils import CustomObjectScope

import layers_builder as layers
from python_utils import utils

# These are the means for the ImageNet pretrained ResNet
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])  # RGB order


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape, weights):
        self.input_shape = input_shape
        if 'pspnet' in weights:
            # Import using model_name
            h5_path = join("weights", "keras", weights + ".h5")
            json_path = join("weights", "keras", weights + ".json")
            self.model = layers.build_pspnet(nb_classes=nb_classes,
                                             resnet_layers=resnet_layers,
                                             input_shape=self.input_shape)

            if os.path.isfile(h5_path):
                print("Keras weights found, loading...")
                self.model.load_weights(h5_path)
            else:
                print("No Keras weights found, import from npy weights.")
                self.set_npy_weights(weights)
        else:
            print('Load pre-trained weights')
            self.model = load_model(weights)

    def predict(self, img, flip_evaluation=False):
        """
        Predict segementation for an image.

        Arguments:
            img: must be rowsxcolsx3
        """
        h_ori, w_ori = img.shape[:2]

        # Preprocess
        img = cv2.resize(img, self.input_shape)
        img = img - DATA_MEAN[:,:,::-1]
        img = img.astype('float32')

        # print("Predicting...")
        probs = self.feed_forward(img, flip_evaluation)

        if img.shape[0:1] != self.input_shape:  # upscale prediction if necessary
            h, w = probs.shape[:2]
            probs = ndimage.zoom(probs, (1. * h_ori / h, 1. * w_ori / w, 1.),
                                 order=1, prefilter=False)

        # print("Finished prediction...")
        return probs

    def feed_forward(self, data, flip_evaluation=False):
        assert data.shape == (self.input_shape[0], self.input_shape[1], 3)

        if flip_evaluation:
            # print("Predict flipped")
            input_with_flipped = np.array(
                [data, np.flip(data, axis=1)])
            prediction_with_flipped = self.model.predict(input_with_flipped)
            prediction = (prediction_with_flipped[
                          0] + np.fliplr(prediction_with_flipped[1])) / 2.0
        else:
            prediction = self.model.predict(np.expand_dims(data, 0))[0]
        return prediction

    def set_npy_weights(self, model_name):
        npy_weights_path = join("weights", "npy", model_name + ".npy")
        json_path = join("weights", "keras", model_name + ".json")
        h5_path = join("weights", "keras", model_name + ".h5")

        print("Importing weights from %s" % npy_weights_path)
        weights = np.load(npy_weights_path, encoding='bytes').item()
        for layer in self.model.layers:
            print(layer.name)
            if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
                mean = weights[layer.name.encode()][
                    'mean'.encode()].reshape(-1)
                variance = weights[layer.name.encode()][
                    'variance'.encode()].reshape(-1)
                scale = weights[layer.name.encode()][
                    'scale'.encode()].reshape(-1)
                offset = weights[layer.name.encode()][
                    'offset'.encode()].reshape(-1)

                self.model.get_layer(layer.name).set_weights(
                    [scale, offset, mean, variance])

            elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
                try:
                    weight = weights[layer.name.encode()]['weights'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight])
                except Exception as err:
                    biases = weights[layer.name.encode()]['biases'.encode()]
                    self.model.get_layer(layer.name).set_weights([weight,
                                                                  biases])
        print('Finished importing weights.')

        print("Writing keras model & weights")
        json_string = self.model.to_json()
        with open(json_path, 'w') as file_handle:
            file_handle.write(json_string)
        self.model.save_weights(h5_path)
        print("Finished writing Keras model & weights")


class PSPNet50(PSPNet):
    """Build a PSPNet based on a 50-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=50,
                        input_shape=input_shape, weights=weights)


class PSPNet101(PSPNet):
    """Build a PSPNet based on a 101-Layer ResNet."""

    def __init__(self, nb_classes, weights, input_shape):
        PSPNet.__init__(self, nb_classes=nb_classes, resnet_layers=101,
                        input_shape=input_shape, weights=weights)

def get_pspnet(model_name, weights):
    if not args.weights:
        if "pspnet50" in model_name:
            pspnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
                              weights=model_name)
        elif "pspnet101" in model_name:
            if "cityscapes" in model_name:
                pspnet = PSPNet101(nb_classes=19, input_shape=(713, 713),
                                   weights=model_name)
            if "voc2012" in model_name:
                pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                                   weights=model_name)

        else:
            print("Network architecture not implemented.")
    else:
        pspnet = PSPNet50(nb_classes=2, input_shape=(
            768, 480), weights=args.weights)
    return pspnet


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

    args = parser.parse_args()

    # Handle input and output args
    images = glob(args.glob_path) if args.glob_path else [args.input_path,]
    images.sort()

    # Directories
    im_dir = os.path.dirname(args.input_path)
    if args.glob_path:
        im_dir = args.glob_path.split('*')[0]
    cm_dir = join(args.output_dir, "cm/")
    pm_dir = join(args.output_dir, "pm/")
    vis_dir = join(args.output_dir, "vis/")


    # Predict
    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        print(args)
        pspnet = get_pspnet(args.model, args.weights)
        for i, img_path in enumerate(images):
            print("Processing image {} / {} : {}".format(i+1,len(images), img_path))
            img = cv2.imread(img_path)

            probs = pspnet.predict(img, args.flip)

            cm = np.argmax(probs, axis=2)
            pm = np.max(probs, axis=2)
            pm = np.array(pm*255, dtype='uint8')

            # color cm is [0.0-1.0] img is [0-255]
            color_cm = utils.add_color(cm)
            alpha_blended = 0.5 * color_cm * 255 + 0.5 * img

            fn = img_path.replace(im_dir, '')
            if fn[0] == '/':
                fn = fn[1:]
            cm_fn = join(cm_dir, fn.replace('.jpg', '.png'))
            pm_fn = join(pm_dir, fn.replace('.jpg', '.png'))
            vis_fn = join(vis_dir, fn)

            if not exists(dirname(cm_fn)):
                os.makedirs(dirname(cm_fn))
            if not exists(dirname(pm_fn)):
                os.makedirs(dirname(pm_fn))
            if not exists(dirname(vis_fn)):
                os.makedirs(dirname(vis_fn))

            cv2.imwrite(cm_fn, cm)
            cv2.imwrite(pm_fn, pm)
            cv2.imwrite(vis_fn, alpha_blended)
