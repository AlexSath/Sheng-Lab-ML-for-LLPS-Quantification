#############################################################
# ADAPTED FROM THIERRY PECOT
# https://github.com/tpecot/NucleiSegmentationAndMarkerIDentification
# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt

### DESCRIPTION ###
# Filename: running_native_cmdline.py
# Purpose command-line interface to run an already trained InceptionV3 model on input directory should contain list of folders with image names, in which a single tif image can be found. In this 'native' implementation, images dimensions are kept the same and directly inputted into the pipeline.
# Author: Alexandre R. Sathler
# Owner: ZuHang Sheng Lab; NIH/NINDS

### COMMAND-LINE PARAMETERS ###
#1 - directory to run model on
#2 - model weights
#3 - output directory for processed images
#4 - only apply on masks ("None")
#5 - number of channels
#6 - number of features / classes
#7 - imaging field x
#8 - imaging field y
#9 - batch size
#10 - normalization ("nuclei segmentation")
############################################################


"""
Import python packages
"""

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import sys
import os
import threading
from threading import Thread, Lock
import h5py
import re
import datetime

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean, rotate, AffineTransform, warp
from skimage.io import imread, imsave
import tifffile as tiff
import matplotlib.pyplot as plt

import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import copy

from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift
from tensorflow.keras.utils import array_to_img, img_to_array, load_img, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.callbacks import TensorBoard

import glob, fnmatch

from models import inceptionV3 as inceptionV3

from helpers import *

import numpy as np
import ipywidgets as widgets
import ipyfilechooser
from ipyfilechooser import FileChooser
from ipywidgets import HBox, Label, Layout


def training(nb_trainings, training_npz, output_dir, nb_channels, nb_classes, imaging_field_x,
             imaging_field_y, learning_rate, nb_epochs, augmentation, batch_size):
    for i in range(nb_trainings):

        model = inceptionV3(n_channels=nb_channels, n_features=nb_classes, dimx=imaging_field_x, dimy=imaging_field_y)

        if augmentation==True:
            model_name = f"InceptionV3_{nb_channels}_ch_" \
                         + f"{nb_classes}_cl_" \
                         + f"{imaging_field_x}_" \
                         + f"{imaging_field_y}_lr_" \
                         + f"{learning_rate}_withDA_" \
                         + f"{nb_epochs}_ep"
        else:
            model_name = f"InceptionV3_{nb_channels}_ch_" \
                         + f"{nb_classes}_cl_" \
                         + f"{imaging_field_x}_" \
                         + f"{imaging_field_y}_lr_" \
                         + f"{learning_rate}_withoutDA_" \
                         + f"{nb_epochs}_ep"

        #return model
        
        train_model_sample(model, training_npz, model_name=model_name, 
                           batch_size = batch_size, n_epoch = nb_epochs, 
                           direc_save = output_dir, lr = learning_rate,
                           augmentation = augmentation)
        del model


"""
Training convnets
"""
def train_model_sample(model, training_data_file_name, model_name = "", batch_size = 32, n_epoch = 100,
    direc_save = "./trained_classifiers/", lr = 0.01, augmentation = True):

    todays_date = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    file_name_save = os.path.join(direc_save, todays_date + "_" + model_name + ".h5")
    logdir = os.path.join("logs", f"{model_name}{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    optimizer = SGD(learning_rate = lr, weight_decay = 1e-7, momentum = 0.9, nesterov = True)
    lr_sched = rate_scheduler(lr = lr, decay = 0.99)

    train_dict, (X_test, Y_test) = get_data_sample(training_data_file_name)
    
    # the data, shuffled and split between train and test sets
    print(train_dict["pixels_x"].shape[0], 'training samples')
    print(X_test.shape[0], 'test samples')

    # determine the number of channels and classes
    input_shape = model.layers[0].output_shape
    n_channels = input_shape[0][-1]
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    # convert class vectors to binary class matrices
    train_dict["labels"] = to_categorical(train_dict["labels"], n_classes)
    Y_test = to_categorical(Y_test, n_classes)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    train_generator = random_sample_generator_centralPixelClassification(
        train_dict["channels"], train_dict["batch"], train_dict["pixels_x"], 
        train_dict["pixels_y"], train_dict["labels"], batch_size, n_channels, 
        n_classes, train_dict["win_x"], train_dict["win_y"], augmentation)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit(train_generator, steps_per_epoch = int(len(train_dict["labels"])/batch_size),
                             epochs = n_epoch, validation_data = (X_test,Y_test),
                             callbacks = [ModelCheckpoint(file_name_save, monitor='val_loss', verbose=0, save_best_only=True,
                                                          mode='auto', save_weights_only=True), LearningRateScheduler(lr_sched),
                                                          tensorboard_callback])


def get_data_sample(file_name):
    training_data = np.load(file_name)
    channels = training_data["channels"]
    batch = training_data["batch"]
    labels = training_data["y"]
    pixels_x, pixels_y = training_data["pixels_x"], training_data["pixels_y"]
    win_x, win_y = training_data["win_x"], training_data["win_y"]
    
    total_batch_size = len(labels)
    num_test = np.int32(np.floor(total_batch_size/10))
    num_train = np.int32(total_batch_size - num_test)
    full_batch_size = np.int32(num_test + num_train)

    """
    Split data set into training data and validation data
    """
    arr = np.arange(len(labels))
    arr_shuff = np.random.permutation(arr)

    train_ind = arr_shuff[0:num_train]
    test_ind = arr_shuff[num_train:full_batch_size]

    X_test, y_test = data_generator(channels.astype("float32"), batch[test_ind], 
                                    pixels_x[test_ind], pixels_y[test_ind], 
                                    labels[test_ind], win_x = win_x, win_y = win_y)
    
    train_dict = {"channels": channels.astype("float32"), "batch": batch[train_ind], 
                  "pixels_x": pixels_x[train_ind], "pixels_y": pixels_y[train_ind], 
                  "labels": labels[train_ind], "win_x": win_x, "win_y": win_y}

    return train_dict, (X_test, y_test)


def data_generator(channels, batch, pixel_x, pixel_y, labels, win_x = 30, win_y = 30):
    img_list = []
    l_list = []
    for b, x, y, l in zip(batch, pixel_x, pixel_y, labels):
        img = channels[b, x-win_x:x+win_x+1, y-win_y:y+win_y+1, :]
        img_list += [img]
        l_list += [l]
    return np.stack(tuple(img_list),axis = 0), np.array(l_list)


def random_sample_generator_centralPixelClassification(img, img_ind, x_coords, y_coords, y_init, batch_size, n_channels, n_classes, win_x, win_y, augmentation = True):

    cpt = 0

    n_images = len(img_ind)
    arr = np.arange(n_images)
    np.random.shuffle(arr)

    while(True):

        # buffers for a batch of data - CHANGED ARS 06/07/2023
        batch_x = np.zeros(tuple([batch_size] + [2*win_x+1, 2*win_y+1] + [n_channels]))
        batch_y = np.zeros(tuple([batch_size] + [n_classes]))
        # get one image at a time
        for k in range(batch_size):

            # get random image
            img_index = arr[cpt%len(img_ind)]

            # open images
            patch_x = img[img_ind[img_index], (x_coords[img_index]-win_x):(x_coords[img_index]+win_x+1), (y_coords[img_index]-win_y):(y_coords[img_index]+win_y+1), :]
            patch_x = np.asarray(patch_x)
            current_class = np.asarray(y_init[img_index])
            current_class = current_class.astype('float32')

            if augmentation:
                augmentationMap = GenerateRandomImgaugAugmentation()
                patch_x = augmentationMap(image=patch_x)


            # save image to buffer
            batch_x[k, :, :, :] = patch_x.astype('float32')
            batch_y[k, :] = current_class
            cpt += 1

        # return the buffer
        yield(batch_x, batch_y)


def GenerateRandomImgaugAugmentation(
        pAugmentationLevel=5,           # number of augmentations
        pEnableFlipping1=True, pEnableFlipping2=True,  # enable x flipping  # enable y flipping        
        pEnableRotation90=True, pEnableRotation=True, pMaxRotationDegree=15, # enable rotation # enable rotation
        pEnableShearX=True, pEnableShearY=True, pMaxShearDegree=15, # enable x shear # enable y shear # maximum shear degree
        pEnableDropOut=True, pMaxDropoutPercentage=0.1, # enable pixel dropout # maximum dropout percentage
        pEnableBlur=True, pBlurSigma=.25, # enable gaussian blur # maximum sigma for gaussian blur
        pEnableSharpness=True, pSharpnessFactor=0.1, # enable sharpness # maximum additional sharpness
        pEnableEmboss=True, pEmbossFactor=0.1, # enable emboss # maximum emboss
        pEnableBrightness=True, pBrightnessFactor=0.1, # enable brightness # maximum +- brightness
        pEnableRandomNoise=True, pMaxRandomNoise=0.1, # enable random noise # maximum random noise strength
        pEnableInvert=True, # enables color invert
        pEnableContrast=True, pContrastFactor=0.1,  # enable contrast change # maximum +- contrast        
):
    
    augmentationMap = []
    augmentationMapOutput = []

    if pEnableFlipping1:
        aug = iaa.Fliplr()
        augmentationMap.append(aug)
        
    if pEnableFlipping2:
        aug = iaa.Flipud()
        augmentationMap.append(aug)

    if pEnableRotation90:
        randomNumber = random.Random().randint(0,3)
        aug = iaa.Rot90(randomNumber)

    if pEnableRotation:
        if random.Random().randint(0, 1)==1:
            randomRotation = random.Random().random()*pMaxRotationDegree
        else:
            randomRotation = -random.Random().random()*pMaxRotationDegree
        aug = iaa.ShearX(randomRotation)
        augmentationMap.append(aug)

    if pEnableShearX:
        if random.Random().randint(0, 1)==1:
            randomShearingX = random.Random().random()*pMaxShearDegree
        else:
            randomShearingX = -random.Random().random()*pMaxShearDegree
        aug = iaa.ShearX(randomShearingX)
        augmentationMap.append(aug)

    if pEnableShearY:
        if random.Random().randint(0, 1)==1:
            randomShearingY = random.Random().random()*pMaxShearDegree
        else:
            randomShearingY = -random.Random().random()*pMaxShearDegree
        aug = iaa.ShearY(randomShearingY)
        augmentationMap.append(aug)

    if pEnableDropOut:
        randomDropOut = random.Random().random()*pMaxDropoutPercentage
        aug = iaa.Dropout(p=randomDropOut, per_channel=False)
        augmentationMap.append(aug)

    if pEnableBlur:
        randomBlur = random.Random().random()*pBlurSigma
        aug = iaa.GaussianBlur(randomBlur)
        augmentationMap.append(aug)

    if pEnableSharpness:
        randomSharpness = random.Random().random()*pSharpnessFactor
        aug = iaa.Sharpen(randomSharpness)
        augmentationMap.append(aug)

    if pEnableEmboss:
        randomEmboss = random.Random().random()*pEmbossFactor
        aug = iaa.Emboss(randomEmboss)
        augmentationMap.append(aug)

    if pEnableBrightness:
        if random.Random().randint(0, 1)==1:
            randomBrightness = 1 - random.Random().random()*pBrightnessFactor
        else:
            randomBrightness = 1 + random.Random().random()*pBrightnessFactor
        aug = iaa.Add(randomBrightness)
        augmentationMap.append(aug)

    if pEnableRandomNoise:
        if random.Random().randint(0, 1)==1:
            randomNoise = 1 - random.Random().random()*pMaxRandomNoise
        else:
            randomNoise = 1 + random.Random().random()*pMaxRandomNoise
        aug = iaa.MultiplyElementwise(randomNoise,  per_channel=True)
        augmentationMap.append(aug)
        
    if pEnableInvert:
        aug = iaa.Invert(1)
        augmentationMap.append(aug)

    if pEnableContrast:
        if random.Random().randint(0, 1)==1:
            randomContrast = 1 - random.Random().random()*pContrastFactor
        else:
            randomContrast = 1 + random.Random().random()*pContrastFactor
        aug = iaa.contrast.LinearContrast(randomContrast)
        augmentationMap.append(aug)

    arr = np.arange(len(augmentationMap))
    np.random.shuffle(arr)
    for i in range(pAugmentationLevel):
        augmentationMapOutput.append(augmentationMap[arr[i]])
    
        
    return iaa.Sequential(augmentationMapOutput)


nb_trainings = 1
training(nb_trainings, sys.argv[1], sys.argv[2], int(sys.argv[3]),
         int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
         float(sys.argv[7]), int(sys.argv[8]), bool(sys.argv[9]),
         int(sys.argv[10]))
