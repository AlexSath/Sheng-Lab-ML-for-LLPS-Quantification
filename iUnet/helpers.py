from tensorflow.keras import backend as K
import os, re
from matplotlib import pyplot as plt
import tifffile as tiff
from skimage.transform import rescale, resize, downscale_local_mean, rotate, AffineTransform, warp
from skimage.io import imread, imsave
import numpy as np
from scipy import ndimage, stats
from cv2 import subtract, erode, dilate, bitwise_not

"""
Helper functions
"""
def categorical_sum(y_true, y_pred):
    return K.sum(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred*0, axis=-1)))

def rate_scheduler(lr = .001, decay = 0.95):
    def output_fn(epoch):
        epoch = int(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

# if mode is input, takes in 4d array with shape (1, x, y, 1)
# if mode is not input, image list should contain images with shape (x, y, z), and 0 <= mode < z
def graph_images(img_names, img_list, output_dir, mode = "input"):
    if len(img_names) != len(img_list):
        ValueError(f"Length of image names is not the same as the number of images!")
    
    cols = 5
    rows = int(np.ceil(len(img_names) / cols))
    fig = plt.figure(figsize = (cols * 3, rows * 3))

    for idx, img, name in zip(list(range(len(img_list))), img_list, img_names):
        ax = fig.add_subplot(rows, cols, idx + 1)
        if mode == "input":
            ax.imshow(img[0,:,:,0])
        else:
            ax.imshow(img[:,:,mode])
        ax.set_title(name, fontsize = 5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"images_{mode}.pdf"))
    

"""
File search and processing tools
"""
def getfiles_keyword(direc_name, channel_name, inverse = False):
    imglist = os.listdir(direc_name)
    imgfiles = []
    if inverse:
        imgfiles = [i for i in imglist if channel_name not in i]
    else:
        imgfiles = [i for i in imglist if channel_name in i]
    
    def sorted_nicely(l):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key = alphanum_key)

    imgfiles = sorted_nicely(imgfiles)
    return imgfiles

def getfiles(direc_name):
    imgfiles = []
    for root, dirs, files in os.walk(direc_name):
        for i in files:
            if ('.png'  in i ) or ('.jpg'  in i ) or ('.tif' in i) or ('.tiff' in i):
                imgfiles.append(os.path.join(root, i))
    imgfiles = sorted_nicely(imgfiles)
    return imgfiles

def get_closest_val(input_val, comp_vals):
    distance = np.max(comp_vals)
    value = 0
    for val in comp_vals:
        this_dist = np.absolute(val - input_val)
        if this_dist < distance:
            distance = this_dist
            value = val
    return value

"""
Image processing tools
"""
def get_images_from_path_list(img_paths, mode = "native"):
    smallest, largest = get_smallest_dim(img_paths), get_largest_dim(img_paths)
    n_channels = len(img_paths)
    all_images = []
    image_names = []

    for stack_iteration in range(len(img_paths)):
        channel_img = get_image(img_paths[stack_iteration])

        #ensure all input images are scaled to uint8 before model feeding
        if channel_img.dtype != np.uint8:
            temp = np.zeros(channel_img.shape, dtype=np.uint8)
            tim = channel_img.astype('float32') / get_tinfo(channel_img).max
            tim *= get_tinfo(temp).max
            temp[:,:] = tim[:,:]
            channel_img = temp

        if mode == "native":
            all_channels = np.zeros((1, channel_img.shape[0], channel_img.shape[1], 1), dtype = 'float32')
            all_channels[0,:,:,0] = channel_img
            
        elif mode == "padded":
            #value = get_closest_val(np.mean(channel_img), (np.iinfo(channel_img.dtype).min, np.iinfo(channel_img.dtype).max))
            value = stats.mode(channel_img.flatten(), keepdims = False)[0]
            all_channels = np.full((1, largest, largest, 1), value, dtype = 'float32')
            if channel_img.size != (largest, largest):
                all_channels[0,:channel_img.shape[0],:channel_img.shape[1],0] = channel_img
            else:
                all_channels[0,:,:,0] = channel_img
    
        elif mode == "resized":
            all_channels = np.zeros((1, smallest, smallest, 1), dtype = 'float32')
            if channel_img.shape != (smallest, smallest):
                channel_img = resize(channel_img, [smallest, smallest])
            all_channels[0,:,:,0] = channel_img
        
        all_images += [all_channels]
        name = os.path.basename(os.path.dirname(img_paths[stack_iteration]))
        image_names.append(name)

    return all_images, image_names


def get_images_from_directory_keyword(data_location, channel_names, inverse = False):
    img_list_channels = []
    for channel in channel_names:
        if inverse:
            img_list_channels += [getfiles_keyword(data_location, channel, True)]
        else:
            img_list_channels += [getfiles_keyword(data_location, channel, False)]

    img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

    n_channels = len(channel_names)
    all_images = []
    image_names = []

    for stack_iteration in range(len(img_list_channels[0])):
        all_channels = np.zeros((1, img_temp.shape[0],img_temp.shape[1], n_channels), dtype = 'float32')
        for j in range(n_channels):
            channel_img = get_image(os.path.join(data_location, img_list_channels[j][stack_iteration]))
            all_channels[0,:,:,j] = channel_img
        all_images += [all_channels]
        image_names.append(os.path.splitext(os.path.basename(img_list_channels[stack_iteration]))[0])

    return all_images, image_names


def process_image_list(images, win_x, win_y, normalization = "nuclei segmentation"):
    processed_images = []
    for idx, img in enumerate(images):
        print(f"Processing image {idx + 1} out of {len(images)}")
        if normalization == "nuclei segmentation":
            for j in range(img.shape[-1]):
                img[0,:,:,j] = process_image(np.float64(img[0,:,:,j]), win_x, win_y)
        elif normalization == "marker identification":
            for j in range(img.shape[-1]):
                img[0,:,:,j] = process_image_onlyLocalAverageSubtraction(np.float64(img[0,:,:,j]), win_x, win_y)
        else: 
            for j in range(img.shape[-1]):
                img[0,:,:,j] = np.float32(img[0,:,:,j])
        processed_images.append(img)
    return processed_images


def process_image(channel_img, win_x, win_y):
    p50 = np.percentile(channel_img, 50)
    channel_img /= max(p50,0.01)
    avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
    channel_img = subtract(channel_img, ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size)
    channel_img[channel_img < 0] = 0
    return channel_img

def process_image_onlyLocalAverageSubtraction(channel_img, win_x, win_y):
    avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
    channel_img = subtract(channel_img, ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size)
    channel_img[channel_img < 0] = 0
    return channel_img

def get_image(file_name):
    if '.tif' in file_name:
        im = tiff.imread(file_name)
    else:
        im = imread(file_name)
    return im

def to8bit(image):
    datatype = get_tinfo(image)
    image = image.astype(np.float32)
    image /= datatype.max
    image *= 255
    return image

def get_image_sizes(data_location):
    width = 256
    height = 256
    nb_channels = 1
    img_list = []
    img_list += [getfiles(data_location)]
    img_temp = get_image(img_list[0][0])
    if len(img_temp.shape)>2:
        if img_temp.shape[0]<img_temp.shape[2]:
            nb_channels = img_temp.shape[0]
            width = img_temp.shape[1]
            height=img_temp.shape[2]
        else:
            nb_channels = img_temp.shape[2]
            width = img_temp.shape[0]
            height=img_temp.shape[1]
    else:
        width = img_temp.shape[0]
        height=img_temp.shape[1]
    return width, height, nb_channels

def get_image_sizes_keyword(data_location, channel_names):
    img_list_channels = []
    for channel in channel_names:       
        img_list_channels += [getfiles_keyword(data_location, channel)]
    img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

    return img_temp.shape

def get_smallest_dim(imglist):
    smallest = 4000
    for img in imglist:
        shape = get_image(img).shape
        if shape[0] < smallest:
            smallest = shape[0]
        if shape[1] < smallest:
            smallest = shape[1]
    return smallest

def get_largest_dim(imglist):
    largest = 0
    for img in imglist:
        shape = get_image(img).shape
        if shape[0] > largest:
            largest = shape[0]
        if shape[1] > largest:
            largest = shape[1]
    return largest

def get_padded(im, size_x, size_y):
    # try:
    #     value = get_closest_val(stats.mode(im, axis=None).mode, 
    #                             (np.finfo(im.dtype).min, np.finfo(im.dtype).max))
    # except Exception as e:
    #     print(e)
    #     value = get_closest_val(stats.mode(im, axis=None).mode, 
    #                             (np.iinfo(im.dtype).min, np.iinfo(im.dtype).max))
    padded = np.full((size_x, size_y), stats.mode(im,axis=None).mode, dtype=im.dtype)
    padded[:im.shape[0],:im.shape[1]] = im
    return padded
    

def get_tinfo(arr_like):
    try:
        return np.iinfo(arr_like.dtype)
    except:
        return np.finfo(arr_like.dtype)
