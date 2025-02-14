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
#1 - path to NPZ file containing training matrix
#2 - path to directory in which model .h5 file will be saved
#3 - number of channels in input images
#4 - number of possible output classes
#5 - patch size x (INT - 65 recommended)
#6 - patch size y (INT - 65 recommended)
#7 - learning rate (FLOAT - 0.01 recommended)
#8 - number of epochs (INT - 10 recommended)
#9 - include data augmentation (BOOL - True recommended)
#10 - batch size (INT - 32 recommended)
#11-n - *weights. (OPTIONAL). 
#       list of integer or float weights for adjustment of categorical crossentropy loss
############################################################


"""
Import python packages
"""

import numpy as np
import tensorflow as tf
from functools import partial

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift
from tensorflow.keras.utils import array_to_img, img_to_array, load_img, to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.python.client import device_lib
from tensorflow import distribute

from models import iunet2

import sys, os, h5py, warnings
import re, datetime, traceback
import glob, fnmatch, copy

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

from helpers import *

def diceloss(y_true, y_pred, weights = [1, 1, 1]):

    def dice_coef(y_true, y_pred, smooth=100):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true * y_pred)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice
        
    # if y_true.shape[-3:] != (128, 128, 3):
    #     raise ValueError(f"y_true has shape {y_true.shape}")

    # for fidx in np.arange(y_true.shape[-1]):

    i1 = dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0])
    i2 = dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1])
    i3 = dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2])
    dice_coefs = [weights[0] * i1, weights[1] * i2, weights[2] * i3]
        
    return  -K.log(K.sum(dice_coefs) / K.cast(K.shape(dice_coefs)[0], tf.float32))

def main():
    global NGPUS
    NGPUS = get_num_gpus()
    nb_trainings = 1
    warnings.showwarning = warn_with_traceback
    try:
        weights = [np.float32(x) for x in sys.argv[11:]]
        weights_sum = np.sqrt(np.sum(np.power(weights, 2)))
        weights2 = [x / weights_sum for x in weights]
    except Exception as e:
        print(traceback.format_exc())
        weights2 = ""
    training(nb_trainings, sys.argv[1], sys.argv[2], int(sys.argv[3]),
                 int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
                 float(sys.argv[7]), int(sys.argv[8]), bool(sys.argv[9]),
                 int(sys.argv[10]), weights2)

        
def get_num_gpus():
    devices = np.array([x.device_type for x in device_lib.list_local_devices()])
    return len(devices[devices == "GPU"])


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def training(nb_trainings, training_npz, output_dir, nb_channels, nb_classes, 
             imaging_field_x, imaging_field_y, learning_rate, nb_epochs, 
             augmentation, batch_size, norm, weights = ""):
    global NGPUS
    global STRATEGY
    global JIT_COMP

    for i in range(nb_trainings):
        input_dims = (imaging_field_x, imaging_field_y, nb_channels)
        
        if NGPUS >= 2:
            JIT_COMP=False
            STRATEGY = distribute.MirroredStrategy()
            with STRATEGY.scope():
                model = iunet2(input_size=input_dims, dropout_rate=0.5, batch_norm=False, n_classes=nb_classes)

        else:
            JIT_COMP=False
            model = iunet2(input_size=input_dims, dropout_rate=0.5, batch_norm=False, n_classes=nb_classes)

        model_name = f"iv3-unet_{nb_channels}-ch_" \
                    + f"{nb_classes}-cl_" \
                    + f"{imaging_field_x}x" \
                    + f"{imaging_field_y}_lr-" \
                    + f"{learning_rate}_{'with' if augmentation else 'without'}DA_" \
                    + f"{norm}_{nb_epochs}-ep"

        #return model
        
        train_model_sample(model, training_npz, weights, model_name=model_name, 
                           batch_size = batch_size, n_epoch = nb_epochs, 
                           direc_save = output_dir, lr = learning_rate,
                           augmentation = augmentation)
        del model


"""
Training convnets
"""
def train_model_sample(model, training_data_file_name, weights, model_name = "", 
                       batch_size=32, n_epoch=100, direc_save="./trained_classifiers/", 
                       lr = 0.01, augmentation = True):
    global NGPUS
    global STRATEGY
    global JIT_COMP

    todays_date = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    file_name_save = os.path.join(direc_save, todays_date + "_" + model_name + ".h5")
    logdir = os.path.join("logs", f"{model_name}{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}")
    tensorboard_callback = TensorBoard(log_dir=logdir)

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

    print(f"input_shape: {input_shape}\n" \
          + f"output_shape: {output_shape}\n" \
          + f"n_channels: {n_channels}\n" \
          + f"n_classes: {n_classes}")

    # convert class vectors to binary class matrices
    #train_dict["labels"] = to_categorical(train_dict["labels"], n_classes)
    #Y_test = to_categorical(Y_test, n_classes)
    
    if type(weights) != type([]) :
        print(f"Training with default Dice Loss")
        if NGPUS >= 2:
            with STRATEGY.scope():
                optimizer = SGD(learning_rate = lr, weight_decay = 1e-7, momentum = 0.9, nesterov = True)
                model.compile(loss=diceloss, optimizer=optimizer, 
                              metrics=['accuracy'], jit_compile=JIT_COMP)
        else:
            optimizer = SGD(learning_rate = lr, weight_decay = 1e-7, momentum = 0.9, nesterov = True)
            model.compile(loss=diceloss, optimizer=optimizer, 
                          metrics=['accuracy'], jit_compile=JIT_COMP)
    else:
        if len(weights) != n_classes:
            raise ValueError(f"Number of weights ({len(weights)})" \
                             + f"must match number of output classes ({n_classes})")
        str_weights = [str(w) for w in weights]
        print(f"Training Dice Loss with weights " \
              + f"{', '.join(str_weights)}")
        dicelossw = partial(diceloss, weights = weights)
        if NGPUS >= 2:
            with STRATEGY.scope():
                optimizer = SGD(learning_rate = lr, weight_decay = 1e-7, momentum = 0.9, nesterov = True)
                model.compile(loss=dicelossw, optimizer=optimizer, 
                              metrics=['accuracy'], jit_compile=JIT_COMP)
        else:
            optimizer = SGD(learning_rate = lr, weight_decay = 1e-7, momentum = 0.9, nesterov = True)
            model.compile(loss=dicelossw, optimizer=optimizer, 
                          metrics=['accuracy'], jit_compile=JIT_COMP)

    train_generator = random_sample_generator(
        train_dict["channels"], train_dict["batch"], train_dict["pixels_x"], 
        train_dict["pixels_y"], train_dict["labels"], batch_size, n_channels, 
        n_classes, train_dict["win_x"], train_dict["win_y"], augmentation
    )

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit(
        train_generator, steps_per_epoch = int(len(train_dict["batch"])/batch_size),
        epochs = n_epoch, validation_data = (X_test,Y_test),
        callbacks = [
            ModelCheckpoint(
                file_name_save, monitor='val_loss', verbose=0, save_best_only=True, 
                mode='auto', save_weights_only=True
            ), LearningRateScheduler(lr_sched), tensorboard_callback,
            EarlyStopping(
                monitor='val_loss', mode='min', min_delta=0.001,
                verbose=1, patience=15
            )
        ]
    )


def get_data_sample(file_name):
    training_data = np.load(file_name)
    
    channels = training_data["channels"]
    assert len(channels[~np.isfinite(channels)]) == 0
    batch = training_data["batch"]
    labels = training_data["y_all"]
    pixels_x, pixels_y = training_data["pixels_x"], training_data["pixels_y"]
    win_x, win_y = int(training_data["win_x"]), int(training_data["win_y"])
    
    total_batch_size = len(batch)
    num_test = np.int32(np.floor(total_batch_size/10))
    num_train = np.int32(total_batch_size - num_test)
    full_batch_size = np.int32(num_test + num_train)

    """
    Split data set into training data and validation data
    """
    arr = np.arange(len(batch))
    arr_shuff = np.random.permutation(arr)

    train_ind = arr_shuff[0:num_train]
    test_ind = arr_shuff[num_train:full_batch_size]

    X_test, y_test = data_generator(channels.astype("float32"), batch[test_ind], 
                                    pixels_x[test_ind], pixels_y[test_ind], 
                                    labels.astype("float32"), win_x = win_x, win_y = win_y)
    
    train_dict = {"channels": channels.astype("float32"), "batch": batch[train_ind], 
                  "pixels_x": pixels_x[train_ind], "pixels_y": pixels_y[train_ind], 
                  "labels": labels.astype("uint8"), "win_x": win_x, "win_y": win_y}

    for key, val in train_dict.items():
        try:
            assert len(val[~np.isfinite(val)]) == 0
        except TypeError as e:
            assert np.isfinite(val) == 1

    return train_dict, (X_test, y_test)


def data_generator(channels, batch, pixel_x, pixel_y, labels, win_x = 30, win_y = 30):
    img_list = []
    l_list = []
    for b, x, y in zip(batch, pixel_x, pixel_y):
        img = channels[b, x-win_x:x+win_x, y-win_y:y+win_y, :]
        l = labels[b, x-win_x:x+win_x, y-win_y:y+win_y, :]
        img_list += [img]
        l_list += [l]
        
    img_stack = np.stack(tuple(img_list), axis = 0).astype('float32')
    l_stack = np.stack(tuple(l_list), axis = 0).astype('float32')
    return img_stack, l_stack


def random_sample_generator(img, img_ind, x_coords, y_coords, y_init, 
                            batch_size, n_channels, n_classes, win_x, 
                            win_y, augmentation = True):

    cpt = 0

    n_images = len(img_ind)
    arr = np.arange(n_images)
    np.random.shuffle(arr)

    while(True):

        # buffers for a batch of data - CHANGED ARS 06/07/2023
        batch_x = np.zeros(
            (batch_size, 2*win_x, 2*win_y, n_channels), 
            dtype='float32')
        batch_y = np.zeros(
            (batch_size, 2*win_x, 2*win_y, n_classes), 
            dtype='float32')
        
        # get one image at a time
        for k in range(batch_size):

            # get random image
            img_index = arr[cpt%len(img_ind)]

            # open images
            patch_x = img[img_ind[img_index], 
                (x_coords[img_index]-win_x):(x_coords[img_index]+win_x),
                (y_coords[img_index]-win_y):(y_coords[img_index]+win_y),
            :]
            patch_x = np.asarray(patch_x)
            
            current_class = y_init[img_ind[img_index],
                (x_coords[img_index]-win_x):(x_coords[img_index]+win_x),
                (y_coords[img_index]-win_y):(y_coords[img_index]+win_y),
            :]
            current_class = np.asarray(current_class)
            current_class = current_class[np.newaxis, :, :, :]

            #if augmentation:
            if augmentation:
                augmentationMap = GenerateRandomImgaugAugmentation(
                    pEnableRandomNoise=True
                )
                patch_x, current_class = augmentationMap(
                    image = patch_x, 
                    segmentation_maps = current_class
                )

            # save image to buffer
            batch_x[k, :, :, :] = patch_x.astype('float32')
            batch_y[k, :, :, :] = current_class.astype('float32')
            cpt += 1

        assert len(batch_x[~np.isfinite(batch_x)]) == 0
        assert len(batch_y[~np.isfinite(batch_y)]) == 0
        assert len(batch_y[(batch_y != 0) & (batch_y != 1)]) == 0
        
        # return the buffer
        yield (batch_x, batch_y)


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
    

if __name__ == "__main__":
    main()
