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
#10 - normalization ("nuclei segmentation", "marker identification", "None")
#11 - implementation ("native", "padded", "resized")
############################################################



from matplotlib import pyplot as plt
from models import iunet, iunet2
import numpy as np
import sys, os
from helpers import *
import tensorflow as tf
import gc
import math


### function RUNNING
# Purpose:
# Pre-Conditions:
# Post-Conditions:
def running(nb_runnings, inputpath, weightpath, outputpath, masks, n_channels,
            n_classes, imgx, imgy, batch_size, norm, imp):
    for i in range(nb_runnings):
        if not os.path.isdir(inputpath):
            sys.exit("Running #"+str(i+1)+": You need to select an input directory for images to be processed")
        if not os.path.isfile(weightpath):
            sys.exit("Running #"+str(i+1)+": You need to select a trained model to process your images")
        if not os.path.isdir(outputpath):
            sys.exit("Running #"+str(i+1)+": You need to select an output directory for processed images")

        print(f"n_features: {n_classes}\n" \
              + f"n_channels: {n_channels}\n" \
              + f"dimx: {imgx}\n" \
              + f"dimy: {imgy}\n" \
              + f"weights_path: {weightpath}")
        
        model = iunet2(input_size=(imgx, imgy, n_channels), 
                       dropout_rate=0.5,
                       batch_norm=False, 
                       n_classes=n_classes,
                       residual=False,
                       weights_path=weightpath)
        
        print(f"data_location: {inputpath}\n" \
              + f"output_location: {outputpath}\n" \
              + f"bs: {batch_size}\n" \
              + f"mask_names: {masks}\n" \
              + f"normalization: {norm}")
        
        image_list = run_models_on_directory(inputpath, outputpath, 
                                model, bs=batch_size, 
                                mask_names=masks, maxDim=256,
                                normalization=norm,
                                imp=imp)
        del model
        return image_list




### function RUN_MODELS_ON_DIRECTORY
# Purpose:
# Pre-Conditions:
# Post-Conditions:
def run_models_on_directory(data_location, output_location, model, bs=32, maxDim=800, 
                            mask_names='None', normalization="nuclei segmentation",
                            imp="native"):

    # determine the number of channels and classes as well as the imaging field dimensions
    input_shape = model.layers[0].output_shape
    n_channels = input_shape[0][1]
    imaging_field_x = int((input_shape[0][1])/2)
    imaging_field_y = int((input_shape[0][2])/2)
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    # determine the image size
    image_size_x, image_size_y, nb_chan = get_image_sizes(data_location)
    print(f"Image dims: ({image_size_x}, {image_size_y})\n" \
          + f"Image Channels: {nb_chan}\n")
    
    # process images
    cpt = 0
    processed_image_list = run_model_on_directory_pixByPix(data_location, output_location, model, 
                                                         win_x=imaging_field_x, win_y=imaging_field_y, bs=bs,
                                                         maxDim=maxDim, normalization=normalization, imp=imp)

    return processed_image_list



### function RUN_MODEL_ON_DIRECTORY_PIXBYPIX
# Purpose:
# Pre-Conditions:
# Post-Conditions:
def run_model_on_directory_pixByPix(data_location, output_location, model, 
                                    win_x = 30, win_y = 30, bs=32, maxDim=800, 
                                    normalization="nuclei segmentation", imp = "native"):
    
    img_paths = getfiles(data_location)
    n_classes = model.layers[-1].output_shape[-1]

    if imp not in ['native', 'padded', 'resized']:
        ValueError(f"Implementation '{imp}' is not recognized.\nOptions are: 'native', 'padded', or 'resized'")
    else:
        image_list, image_names = get_images_from_path_list(img_paths, mode=imp)

    image_list = process_image_list(image_list, win_x, win_y, normalization)
    graph_images(image_names, image_list, output_location)

    predicted_image_list = []
    for idx, img in enumerate(image_list):
        print(f"Processing image {idx + 1} of {len(image_list)}: {image_names[idx]} ", end=' ')

        # get raw image dimensions for decropping of the image)
        img_dims = tiff.imread(os.path.join(data_location, image_names[idx], os.path.basename(img_paths[idx]))).shape

        image_prediction = run_model_pixByPix(img, model, win_x=win_x, win_y = win_y, bs=bs, maxDim=maxDim, normalization=normalization)

        #largest = get_largest_dim(img_paths)
        #image_prediction = np.full((largest, largest, 3), 255, dtype=np.float32)
        
        image_prediction = image_prediction[win_x:img_dims[0]+win_x,win_y:img_dims[1]+win_y,:]

        top_vals, _ = tf.math.top_k(image_prediction, 1)
        image_prediction = tf.cast(tf.greater_equal(image_prediction, top_vals), tf.float32).numpy()
        predicted_image_list += [image_prediction]

        # Save images
        for i in range(n_classes):
            cnnout_dir = os.path.join(output_location, image_names[idx])
            if not os.path.isdir(cnnout_dir):
                os.mkdir(cnnout_dir)
            cnnout_name = os.path.join(cnnout_dir, f"image_c{i}.tif")
            tiff.imwrite(cnnout_name, image_prediction[:,:,i])

    for i in range(n_classes):
        graph_images(image_names, predicted_image_list, output_location, i)
    
    return predicted_image_list



### function RUN_MODEL_PIXBYPIX
# Purpose:
# Pre-Conditions:
# Post-Conditions:
def run_model_pixByPix(img, model, win_x = 30, win_y = 30, std = False, split = True, process = True, bs=32, 
                       maxDim=256, normalization = "nuclei segmentation", imp = "native"):                           

    image_size_x = img.shape[1]
    image_size_y = img.shape[2]
    input_padded = np.zeros((img.shape[0], image_size_x + 2 * win_x, image_size_y + 2 * win_y, img.shape[-1]), dtype = np.float32)
    for i in range(img.shape[-1]):
        img_temp = np.pad(img[0,:,:,i], pad_width = [(win_x, win_x), (win_y, win_y)], mode = 'reflect')
        input_padded[0,:,:,i] = img_temp[:, :]
    print(f"size={input_padded.shape}")
    img = input_padded
    n_classes = model.layers[-1].output_shape[-1]
    
    model_output = np.zeros((image_size_x, image_size_y, n_classes), dtype = np.float32)

    print(f"Input image size: {image_size_x}x{image_size_y}\n" \
          + f"Padded image size: {img.shape[1]}x{img.shape[2]}\n" \
          + f"Stiched model output size: {model_output.shape[0]}x{model_output.shape[1]}")
        
    x_minIterator, y_minIterator = win_x, win_y
    x_maxIterator = min(image_size_x, maxDim) - win_x
    y_maxIterator = min(image_size_y, maxDim) - win_y
    xadj, yadj = int(win_x), int(win_y)

    count = 1
    while x_minIterator<(image_size_x-win_x) and y_minIterator<(image_size_y-win_y):
        print(f"Iteration {count}/" \
            + f"{math.ceil((image_size_x-win_x)/maxDim+1) * math.ceil((image_size_y-win_y)/maxDim+1)}", end = "\t")
        #print(x_minIterator, x_maxIterator, y_minIterator, y_maxIterator)
        #print(f"Generating test images... ", end=' ')
        test_images = []
        for x in range(x_minIterator, x_maxIterator):
            for y in range(y_minIterator, y_maxIterator):
                this_img = img[0, x-xadj:x+xadj, y-yadj:y+yadj, :]
                test_images.append(this_img)
               
        test_images = np.asarray(test_images)
        #print(f"Allocating a set size {test_images.shape}...")
        test_images = tf.cast(tf.constant(test_images), tf.float32)

        #print(f"Predicting on a set of size {test_images.shape}...")
        predictions = model.predict(test_images, verbose=2, batch_size=bs)

        cpt = 0
        #print(f"Appending to model output...", end=' ')
        for x in range(x_minIterator, x_maxIterator):
            for y in range(y_minIterator, y_maxIterator):
                assert predictions[cpt,:].shape == (96, 96, 3)
                model_output[x-xadj:x+xadj,y-yadj:y+yadj,:] = predictions[cpt,:]
                cpt += 1

        gc.collect()

        #print(f"Adjusting Iterator...")
        if x_maxIterator < image_size_x-xadj:
            x_minIterator = min(x_maxIterator,image_size_x)
            if image_size_x-x_minIterator < maxDim:
                x_maxIterator = image_size_x-xadj
            else:
                x_maxIterator = x_minIterator+maxDim-xadj
        else:       
            x_minIterator = xadj
            x_maxIterator = min(image_size_x,maxDim)-xadj
            y_minIterator = min(y_maxIterator,image_size_y)
            if image_size_y-y_minIterator < maxDim:
                y_maxIterator = image_size_y-yadj
            else:
                y_maxIterator = y_minIterator+maxDim-yadj

        count += 1

    return model_output



#################################
### CALLING THE MAIN FUNCTION ###
#################################
if __name__ == "__main__":
    nb_trainings = 1
    running(nb_trainings, sys.argv[1], sys.argv[2], sys.argv[3],
             sys.argv[4], int(sys.argv[5]), int(sys.argv[6]),
             int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]),
             sys.argv[10], sys.argv[11]) 
