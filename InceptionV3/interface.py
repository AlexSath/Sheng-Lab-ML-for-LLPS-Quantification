"""
Import python packages
"""

import numpy as np
import os, re, glob, fnmatch

from helpers import *

import numpy as np
import ipywidgets as widgets
import ipyfilechooser
from ipyfilechooser import FileChooser
from ipywidgets import HBox, Label, Layout

"""
Interfaces
"""
def data_preprocessing_interface(nb_trainings, directory):
    input_dir = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    nb_classes = np.zeros([nb_trainings], HBox)
    window_size_x = np.zeros([nb_trainings], HBox)
    window_size_y = np.zeros([nb_trainings], HBox)
    normalization = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input directory")
        input_dir[i] = FileChooser(directory)
        display(input_dir[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser(f'{directory}Npz')
        display(output_dir[i])

        label_layout = Layout(width='230px',height='30px')

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        window_size_x[i] = HBox([Label('Half window size for imaging field in x:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(window_size_x[i])

        window_size_y[i] = HBox([Label('Half window size for imaging field in y:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(window_size_y[i])

        normalization[i] = HBox([Label('Normalization:', layout=label_layout), widgets.RadioButtons(
            options=['nuclei segmentation', 'marker identification'],description='', disabled=False)])
        display(normalization[i])

    parameters.append(input_dir)
    parameters.append(output_dir)
    parameters.append(nb_classes)
    parameters.append(window_size_x)
    parameters.append(window_size_y)
    parameters.append(normalization)
    
    return parameters  

def training_parameters_interface(nb_trainings, directory):
    training_dir = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    learning_rate = np.zeros([nb_trainings], HBox)
    nb_epochs = np.zeros([nb_trainings], HBox)
    augmentation = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training dataset")
        training_dir[i] = FileChooser(directory)
        display(training_dir[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser(directory)
        display(output_dir[i])

        label_layout = Layout(width='200px',height='30px')

        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        imaging_field_x[i] = HBox([Label('Imaging field in x:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_x[i])

        imaging_field_y[i] = HBox([Label('Imaging field in y:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_y[i])

        learning_rate[i] = HBox([Label('Learning rate:', layout=label_layout), widgets.FloatText(
            value=1e-2, description='', disabled=False)])
        display(learning_rate[i])

        nb_epochs[i] = HBox([Label('Number of epochs:', layout=label_layout), widgets.IntText(
            value=10, description='', disabled=False)])
        display(nb_epochs[i])

        augmentation[i] = HBox([Label('Augmentation:', layout=label_layout), widgets.Checkbox(
            value=True, description='', disabled=False)])
        display(augmentation[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(batch_size[i])

    parameters.append(training_dir)
    parameters.append(output_dir)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
    parameters.append(learning_rate)
    parameters.append(nb_epochs)
    parameters.append(augmentation)
    parameters.append(batch_size)
    
    return parameters  

def transfer_learning_parameters_interface(nb_trainings, directory):
    training_dir = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    pretrained_model = np.zeros([nb_trainings], FileChooser)
    nb_classes_pretrained_model = np.zeros([nb_trainings], FileChooser)
    last_layer_training = np.zeros([nb_trainings], HBox)
    nb_epochs_last_layer = np.zeros([nb_trainings], HBox)
    learning_rate_last_layer = np.zeros([nb_trainings], HBox)
    all_network_training = np.zeros([nb_trainings], HBox)
    nb_epochs_all = np.zeros([nb_trainings], HBox)
    learning_rate_all = np.zeros([nb_trainings], HBox)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    augmentation = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training dataset")
        training_dir[i] = FileChooser(directory)
        display(training_dir[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser(directory)
        display(output_dir[i])
        print('\x1b[1m'+"Pretrained model")
        pretrained_model[i] = FileChooser(directory)
        display(pretrained_model[i])

        label_layout = Layout(width='250px',height='30px')

        nb_classes_pretrained_model[i] = HBox([Label('Number of classes in the pretrained model:', layout=label_layout), widgets.IntText(
            value=3, description='',disabled=False)])
        display(nb_classes_pretrained_model[i])

        last_layer_training[i] = HBox([Label('Training last layer only first:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(last_layer_training[i])

        nb_epochs_last_layer[i] = HBox([Label('Number of epochs for last_layer training:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_epochs_last_layer[i])

        learning_rate_last_layer[i] = HBox([Label('Learning rate for last_layer training:', layout=label_layout), widgets.FloatText(
            value=0.05, description='', disabled=False)])
        display(learning_rate_last_layer[i])

        all_network_training[i] = HBox([Label('Training all network:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(all_network_training[i])

        nb_epochs_all[i] = HBox([Label('Number of epochs for all network training:', layout=label_layout), widgets.IntText(
            value=5, description='', disabled=False)])
        display(nb_epochs_all[i])

        learning_rate_all[i] = HBox([Label('Learning rate for all network training:', layout=label_layout), widgets.FloatText(
            value=0.01, description='', disabled=False)])
        display(learning_rate_all[i])

        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        imaging_field_x[i] = HBox([Label('Imaging field in x:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_x[i])

        imaging_field_y[i] = HBox([Label('Imaging field in y:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_y[i])

        augmentation[i] = HBox([Label('Augmentation:', layout=label_layout), widgets.Checkbox(
            value=True, description='', disabled=False)])
        display(augmentation[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(batch_size[i])

    parameters.append(training_dir)
    parameters.append(output_dir)
    parameters.append(pretrained_model)
    parameters.append(nb_classes_pretrained_model)
    parameters.append(last_layer_training)
    parameters.append(nb_epochs_last_layer)
    parameters.append(learning_rate_last_layer)
    parameters.append(all_network_training)
    parameters.append(nb_epochs_all)
    parameters.append(learning_rate_all)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
    parameters.append(augmentation)
    parameters.append(batch_size)
    
    return parameters  

def running_parameters_interface(nb_trainings, directory):
    input_dir = np.zeros([nb_trainings], FileChooser)
    input_classifier = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    input_masks = np.zeros([nb_trainings], FileChooser)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    normalization = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input directory")
        input_dir[i] = FileChooser(directory)
        display(input_dir[i])
        print('\x1b[1m'+"Input model")
        input_classifier[i] = FileChooser(directory)
        display(input_classifier[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser(directory)
        display(output_dir[i])

        label_layout = Layout(width='150px',height='30px')

        input_masks[i] = HBox([Label('Only apply on masks:', layout=label_layout), widgets.Text(
            value='None', description='', disabled=False)])
        display(input_masks[i])

        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        imaging_field_x[i] = HBox([Label('Imaging field in x:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_x[i])

        imaging_field_y[i] = HBox([Label('Imaging field in y:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_y[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(batch_size[i])

        normalization[i] = HBox([Label('Normalization:', layout=label_layout), widgets.RadioButtons(
            options=['nuclei segmentation', 'marker identification'],description='', disabled=False)])
        display(normalization[i])

    parameters.append(input_dir)
    parameters.append(input_classifier)
    parameters.append(output_dir)
    parameters.append(input_masks)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
    parameters.append(batch_size)
    parameters.append(normalization)
    
    return parameters