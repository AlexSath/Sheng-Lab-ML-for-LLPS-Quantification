from keras.models import Model
from keras.layers import *
from switchnorm import SwitchNormalization  
import tensorflow as tf
import numpy as np


def _common_spectral_pool(images, filter_size):
    assert len(images.get_shape().as_list()) == 4
    #print(images.get_shape().as_list())
    assert filter_size >= 3
    if filter_size % 2 == 1:
        n = int((filter_size-1)/2)
        top_left = images[:, :n+1, :n+1, :]
        #print(top_left)
        top_right = images[:, :n+1, -n:, :]
        #print(top_right)
        bottom_left = images[:, -n:, :n+1, :]
        #print(bottom_left)
        bottom_right = images[:, -n:, -n:, :]
        #print(bottom_right)
        top_combined = tf.concat([top_left, top_right], axis=-2)
        #print(top_combined)
        bottom_combined = tf.concat([bottom_left, bottom_right], axis=-2)
        #print(bottom_combined)
        all_together = tf.concat([top_combined, bottom_combined], axis=1)
        #print(all_together)
    else:
        n = filter_size // 2
        top_left = images[:, :n, :n, :]
        #print(top_left)
        top_middle = tf.expand_dims(
            tf.cast(0.5 ** 0.5, tf.complex64) *
            (images[:, :n, n, :] + images[:, :n, -n, :]),
            -2
        )
        #print(top_middle)
        top_right = images[:, :n, -(n-1):, :]
        #print(top_right)
        middle_left = tf.expand_dims(
            tf.cast(0.5 ** 0.5, tf.complex64) *
            (images[:, n, :n, :] + images[:, -n, :n, :]),
            -3
        )
        #print(middle_left)
        middle_middle = tf.expand_dims(
            tf.expand_dims(
                tf.cast(0.5, tf.complex64) *
                (images[:, n, n, :] + images[:, n, -n, :] +
                 images[:, -n, n, :] + images[:, -n, -n, :]),
                -2
            ),
            -2
        )
        #print(middle_middle)
        middle_right = tf.expand_dims(
            tf.cast(0.5 ** 0.5, tf.complex64) *
            (images[:, n, -(n-1):, :] + images[:, -n, -(n-1):, :]),
            -3
        )
        #print(middle_right)
        bottom_left = images[:, -(n-1):, :n, :]
        #print(bottom_left)
        bottom_middle = tf.expand_dims(
            tf.cast(0.5 ** 0.5, tf.complex64) *
            (images[:, -(n-1):, n, :] + images[:, -(n-1):, -n, :]),
            -2
        )
        #print(bottom_middle)
        bottom_right = images[:, -(n-1):, -(n-1):, :]
        #print(bottom_right)
        top_combined = tf.concat(
            [top_left, top_middle, top_right],
            axis=-2
        )
        #print(top_combined)
        middle_combined = tf.concat(
            [middle_left, middle_middle, middle_right],
            axis=-2
        )
        #print(middle_combined)
        bottom_combined = tf.concat(
            [bottom_left, bottom_middle, bottom_right],
            axis=-2
        )
        #print(bottom_combined)
        all_together = tf.concat(
            [top_combined, middle_combined, bottom_combined],
            axis=-3
        )
        #print(all_together)
    return all_together

# class SpectralPoolLayer(keras.layers.Layer):
#     def __init__(self, input_x, filter_size=3,
#                  freq_dropout_lower_bound=None,
#                  freq_dropout_upper_bound=None,
#                  activation=tf.nn.relu, m=0,
#                  train_phase=False):
#         super().__init__()
#         self.im_fft = tf.signal.fft2d(tf.cast(input_x, tf.complex64))
#         self.im_transformed = _common_spectral_pool(self.im_fft, filter_size)

#         if (freq_dropout_lower_bound is not None and
#             freq_dropout_upper_bound is not None):
#             self.im_downsampled = tf.cond(
#                     train_phase,
#                     true_fn=self.true_fn,
#                     false_fn=self.false_fn)
#         else:
#             self.im_out = tf.math.real(
#                 tf.signal.ifft2d(self.im_transformed))

#         if activation is not None:
#             cell_out = activation(im_out)
#         else:
#             cell_out = im_out

#         tf.summary.histogram(f'sp_layer/{cell_out}/activation')
#         self.cell_out = cell_out
            

#     def true_fn():
#         tf_random_cutoff = tf.random_uniform([],
#                         freq_dropout_lower_bound,
#                         freq_dropout_upper_bound)
#         dropout_mask = _frequency_dropout_mask(
#                         filter_size,
#                         tf_random_cutoff)
#         return self.im_transformed * dropout_mask

#     def false_fn():
#         return self.im_transformed

#     def output(self):
#         return self.cell_out
        
        

class spectral_pool_layer(object):

    def __init__(
        self,
        input_x,
        filter_size=3,
        freq_dropout_lower_bound=None,
        freq_dropout_upper_bound=None,
        activation=tf.nn.relu,
        m=0,
        train_phase=False
    ):
        
        # assert only 1 dimension passed for filter size
        assert isinstance(filter_size, int)

        input_shape = input_x.get_shape().as_list()
    
        assert len(input_shape) == 4
        _, H, W, _ = input_shape
        #_, _, H, W = input_shape
        assert H == W
        

        with tf.compat.v1.variable_scope('spectral_pool_layer_{0}'.format(m)):
            im_fft = tf.signal.fft2d(tf.cast(input_x, tf.complex64))
            im_transformed = _common_spectral_pool(im_fft, filter_size)
            if (
                freq_dropout_lower_bound is not None and
                freq_dropout_upper_bound is not None
            ):
                def true_fn():
                    tf_random_cutoff = tf.random_uniform(
                        [],
                        freq_dropout_lower_bound,
                        freq_dropout_upper_bound
                    )
                    dropout_mask = _frequency_dropout_mask(
                        filter_size,
                        tf_random_cutoff
                    )
                    return im_transformed * dropout_mask

                def false_fn():
                    return im_transformed

                im_downsampled = tf.cond(
                    train_phase,
                    true_fn=true_fn,
                    false_fn=false_fn
                )
                
            else:
                im_out = tf.math.real(tf.signal.ifft2d(im_transformed))

            if activation is not None:
                cell_out = activation(im_out)
            else:
                cell_out = im_out
            tf.summary.histogram('sp_layer/{}/activation'.format(m), cell_out)

        self.cell_out = cell_out

    def output(self):
        return self.cell_out

def get_spectral_pool_layer_same(input_x):
    channel = int(input_x.shape[-1])
    kernel_size = int(input_x.shape[1])
    spl = spectral_pool_layer(input_x=input_x,
    filter_size=kernel_size,
    freq_dropout_lower_bound=None,
    freq_dropout_upper_bound=None,
    activation=tf.nn.relu,
    m=0,
    train_phase=True)
    return spl.cell_out

def get_spectral_pool_layer_valid(input_x):
    channel = int(input_x.shape[-1])
    kernel_size = int(input_x.shape[1])
    spl = spectral_pool_layer(input_x=input_x,
    filter_size=int(kernel_size/2),
    freq_dropout_lower_bound=None,
    freq_dropout_upper_bound=None,
    activation=tf.nn.relu,
    m=0,
    train_phase=True)
    return spl.cell_out

def get_spectral_pool_layer_short(input_x):
    channel = int(input_x.shape[-1])
    kernel_size = int(input_x.shape[1])
    spl = spectral_pool_layer(input_x=input_x,
    filter_size=int(kernel_size/16),
    freq_dropout_lower_bound=None,
    freq_dropout_upper_bound=None,
    activation=tf.nn.relu,
    m=0,
    train_phase=True)
    return spl.cell_out

def hybrid_pool_layer_same(pool_size=(2, 2)):
    def apply(x):
        channel = int(x.shape[-1])
        kernel_size = int(x.shape[1])
        poolout = MaxPooling2D(pool_size, padding='same', strides=1)(x)
        spectral_pool = Lambda(get_spectral_pool_layer_same,
                               output_shape=(poolout.shape[1], poolout.shape[1], channel))(x)
        assert spectral_pool.shape == poolout.shape
        return Conv2D(int(x.shape[-1]), (1, 1))(
            Concatenate()([poolout, spectral_pool])
        )
    return apply

def hybrid_pool_layer_valid(pool_size=(2, 2)):
    def apply(x):
        channel = int(x.shape[-1])
        kernel_size = int(x.shape[1])
        poolout = MaxPooling2D(pool_size)(x)
        spectral_pool = Lambda(get_spectral_pool_layer_valid,
                               output_shape=(poolout.shape[1], poolout.shape[1], channel))(x)
        assert spectral_pool.shape == poolout.shape
        return Conv2D(int(x.shape[-1]), (1, 1))(
            Concatenate()([poolout, spectral_pool])
        )
    return apply

def hybrid_pool_layer_short(pool_size=(16, 16)):
    def apply(x):
        channel = int(x.shape[-1])
        kernel_size = int(x.shape[1])
        poolout = MaxPooling2D(pool_size)(x)
        spectral_pool = Lambda(get_spectral_pool_layer_short,
                               output_shape=(poolout.shape[1], poolout.shape[1], channel))(x)
        assert spectral_pool.shape == poolout.shape
        return Conv2D(int(x.shape[-1]), (1, 1))(
            Concatenate()([poolout, spectral_pool])
        )
    return apply



def inception_conv_layer(kernels, dropout, batch_norm=False, residual=False):
    def apply(x):
        axis = 3
        conv1 = Conv2D(kernels, (3, 3), padding='same')(x)
        conv2 = Conv2D(kernels, (1, 1), padding='same')(x)
        conv3 = Conv2D(kernels, (5, 5), padding='same')(x)
        pool = hybrid_pool_layer_same()(x)
        
        if batch_norm is True:
            conv1 = BatchNormalization(axis=axis)(conv1)
            conv2 = BatchNormalization(axis=axis)(conv2)
            conv3 = BatchNormalization(axis=axis)(conv3)
            pool = BatchNormalization(axis=axis)(pool)
        conv1 = Activation('relu')(conv1)
        conv2 = Activation('relu')(conv2)
        conv3 = Activation('relu')(conv3)
        pool = Activation('relu')(pool)
        
        concat_1 = concatenate([conv1,conv2,conv3,pool],axis=-1)
        f_conv = Conv2D(kernels, (1, 1), padding='same', activation='relu')(concat_1)
        
        if dropout > 0:
            f_conv = Dropout(dropout)(f_conv)
        
        if residual:
            shortcut = Conv2D(kernels, kernel_size=(1, 1), padding='same')(x)
            if batch_norm is True:
                shortcut = BatchNormalization(axis=axis)(shortcut)
    
            res_path = add([shortcut, f_conv])
            return res_path
        return f_conv
    return apply


def iunet(input_size=(128,128,1), dropout_rate=0.0, batch_norm=False, n_classes=1, residual=False):
    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]  
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL))
    s = SwitchNormalization(axis=-1) (inputs)

    c10 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same') (s)
    c11 = Conv2D(16, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (s)
    c12 = Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same') (s)
    p1 = hybrid_pool_layer_same()(s) #output_shape = (128,128,3)
    p1 = concatenate([c11,c10,c12,p1])
    p1 = Conv2D(16, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (p1)
    p1 = Dropout(0.1) (p1)
    
    r1 = hybrid_pool_layer_valid()(p1) #output_shape=(64,64,16)
    
    c20 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same') (r1)
    c21 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (r1)
    c22 = Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same') (r1)
    p2 = hybrid_pool_layer_same()(r1) #output_shape=(64, 64, 16)
    p2 = concatenate([c21,c20,c22,p2])
    p2 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (p2)
    p2 = Dropout(0.1) (p2)
    
    r2 = hybrid_pool_layer_valid()(p2) #output_shape=(64, 64, 16)
    
    c30 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same') (r2)
    c31 = Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (r2)
    c32 = Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same') (r2)
    p3 = hybrid_pool_layer_same()(r2) #output_shape=(32, 32, 32)
    p3 = concatenate([c31,c30,c32,p3])
    p3 = Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (p3)
    p3 = Dropout(0.2) (p3)
    
    r3 = hybrid_pool_layer_valid()(p3) #output_shape=(16, 16, 64)
    
    c40 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same') (r3)
    c41 = Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (r3)
    c42 = Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same') (r3)
    p4 = hybrid_pool_layer_same()(r3) #output_shape=(16, 16, 64)
    p4 = concatenate([c41,c40,c42,p4])
    p4 = Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (p4)
    p4 = Dropout(0.2) (p4)
    
    r4 = hybrid_pool_layer_valid()(p4) #output_shape=(8, 8, 128)
    
    c50 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same') (r4)
    c51 = Conv2D(256, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (r4)
    c52 = Conv2D(256, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same') (r4)
    p5 = hybrid_pool_layer_same()(r4) #output_shape=(8, 8, 128)
    sc = hybrid_pool_layer_short()(s) #output_shape=(8, 8, 3)
    r5 = concatenate([c51,c50,c52,p5,sc])
    r5 = Conv2D(256, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (r5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (r5)
    u6 = concatenate([u6, p4])
    c60 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same') (u6)
    c61 = Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (u6)
    c62 = Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same') (u6)
    p6 = hybrid_pool_layer_same()(u6) #output_shape=(16, 16, 256)
    r6 = concatenate([c61,c60,c62,p6])
    r6 = Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (r6)
    r6 = Dropout(0.2) (r6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (r6)
    u7 = concatenate([u7, p3])
    c70 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same') (u7)
    c71 = Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (u7)
    c72 = Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same') (u7)
    p7 = hybrid_pool_layer_same()(u7) #output_shape=(32, 32, 128)
    r7 = concatenate([c71,c70,c72,p7])
    r7 = Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (r7)
    r7 = Dropout(0.2) (r7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (r7)
    u8 = concatenate([u8, p2])
    c80 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same') (u8)
    c81 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (u8)
    c82 = Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same') (u8)
    p8 = hybrid_pool_layer_same()(u8) #output_shape=(64, 64, 64)
    r8 = concatenate([c81,c80,c82,p8])
    r8 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (r8)
    r8 = Dropout(0.1) (r8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (r8)
    u9 = concatenate([u9, p1], axis=3)
    c90 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same') (u9)
    c91 = Conv2D(16, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (u9)
    c92 = Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same') (u9)
    p9 = hybrid_pool_layer_same()(u9) #output_shape=(128, 128, 32)
    r9 = concatenate([c91,c90,c92,p9])
    r9 = Conv2D(16, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same') (r9)
    r9 = Dropout(0.1) (r9)

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid') (r9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
    

def iunet2(input_size=(128,128,3), dropout_rate=0.0, batch_norm=False, n_classes=1, residual=False, weights_path=None):

    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]  
    KERNELS = int(INPUT_SIZE / 8)
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL), dtype=tf.float32)
    axis = 3
    
    s = SwitchNormalization(axis=-1) (inputs)
    
    # CONTRACTION PHASE
    conv_128 = inception_conv_layer(KERNELS, dropout_rate, batch_norm, residual) (s)
    pool_64 = hybrid_pool_layer_valid()(conv_128)
    
    conv_64 = inception_conv_layer(2*KERNELS, dropout_rate, batch_norm, residual) (pool_64)
    pool_32 = hybrid_pool_layer_valid()(conv_64)
    
    conv_32 = inception_conv_layer(4*KERNELS, dropout_rate, batch_norm, residual) (pool_32)
    pool_16 = hybrid_pool_layer_valid()(conv_32)
    
    conv_16 = inception_conv_layer(8*KERNELS, dropout_rate, batch_norm, residual) (pool_16)
    pool_8 = hybrid_pool_layer_valid()(conv_16)
    
    ## Bottleneck Layer ##
    c50 = Conv2D(16*KERNELS, (3, 3), padding='same') (pool_8)
    c51 = Conv2D(16*KERNELS, (1, 1), padding='same') (pool_8)
    c52 = Conv2D(16*KERNELS, (5, 5), padding='same') (pool_8)
    p5 = hybrid_pool_layer_same() (pool_8) #output_shape=(8, 8, 128)
    sc = hybrid_pool_layer_short() (s) #output_shape=(8, 8, 3)
    if batch_norm is True:
        c50 = BatchNormalization(axis=axis)(c50)
        c51 = BatchNormalization(axis=axis)(c51)
        c52 = BatchNormalization(axis=axis)(c52)
        p5 = BatchNormalization(axis=axis)(p5)
        sc = BatchNormalization(axis=axis)(sc)
    c50 = Activation('relu')(c50)
    c51 = Activation('relu')(c51)
    c52 = Activation('relu')(c52)
    p5 = Activation('relu')(p5)
    sc = Activation('relu')(sc)
    r5 = concatenate([c51,c50,c52,p5,sc])
    r5 = Conv2D(16*KERNELS, (1, 1), activation='relu', padding='same') (r5)
    
    ## Expansion Phase ##
    uconv_16 = Conv2DTranspose(8*KERNELS, (2, 2), strides=(2, 2), padding='same') (r5)
    uconv_16 = concatenate([uconv_16, conv_16])
    uconv_16 = inception_conv_layer(8*KERNELS, dropout_rate, batch_norm, residual) (uconv_16)
    
    uconv_32 = Conv2DTranspose(4*KERNELS, (2, 2), strides=(2, 2), padding='same') (uconv_16)
    uconv_32 = concatenate([uconv_32, conv_32])
    uconv_32 = inception_conv_layer(4*KERNELS, dropout_rate, batch_norm, residual) (uconv_32)
    
    uconv_64 = Conv2DTranspose(2*KERNELS, (2, 2), strides=(2, 2), padding='same') (uconv_32)
    uconv_64 = concatenate([uconv_64, conv_64])
    uconv_64 = inception_conv_layer(2*KERNELS, dropout_rate, batch_norm, residual) (uconv_64)
    
    uconv_128 = Conv2DTranspose(KERNELS, (2, 2), strides=(2, 2), padding='same') (uconv_64)
    uconv_128 = concatenate([uconv_128, conv_128])
    uconv_128 = inception_conv_layer(KERNELS, dropout_rate, batch_norm, residual) (uconv_128)
    
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid') (uconv_128)
    
    model = Model(inputs=[inputs], outputs=[outputs])

    if weights_path != None:
        model.load_weights(weights_path)
    
    return model


def iunet64(input_size=(64, 64, 1), dropout_rate=0.0, batch_norm=False, n_classes=1, residual=False, weights = False):

    INPUT_SIZE = input_size[0]
    INPUT_CHANNEL = input_size[-1]
    KERNELS = INPUT_SIZE / 16
    
    inputs = Input((INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL), dtype=tf.float32)
    axis = 3
    
    s = SwitchNormalization(axis=-1) (inputs)
    
    # CONTRACTION PHASE
    conv_128 = inception_conv_layer(KERNELS, dropout_rate, batch_norm, residual)(s)
    pool_64 = hybrid_pool_layer_valid()(conv_128)
    
    conv_64 = inception_conv_layer(2*KERNELS, dropout_rate, batch_norm, residual)(pool_64)
    pool_32 = hybrid_pool_layer_valid()(conv_64)
    
    conv_32 = inception_conv_layer(4*KERNELS, dropout_rate, batch_norm, residual)(pool_32)
    pool_16 = hybrid_pool_layer_valid()(conv_32)
    
    #conv_16 = inception_conv_layer(8*KERNELS, dropout_rate, batch_norm, residual)(pool_16)
    #pool_8 = hybrid_pool_layer_valid()(conv_16)
    
    ## Bottleneck Layer ##
    # for 128 x 128 model, requires 16*KERNELS instead of current 8*KERNELS
    c50 = Conv2D(8*KERNELS, (3, 3), padding='same') (pool_16)#(pool_8)
    c51 = Conv2D(8*KERNELS, (1, 1), padding='same') (pool_16)#(pool_8)
    c52 = Conv2D(8*KERNELS, (5, 5), padding='same') (pool_16)#(pool_8)
    p5 = hybrid_pool_layer_same()(pool_16) #output_shape=(8, 8, 128)
    sc = hybrid_pool_layer_short((8,8)) (s) #output_shape=(8, 8, 3); added (8,8) argument for 64x64 input
    if batch_norm is True:
        c50 = BatchNormalization(axis=axis)(c50)
        c51 = BatchNormalization(axis=axis)(c51)
        c52 = BatchNormalization(axis=axis)(c52)
        p5 = BatchNormalization(axis=axis)(p5)
        sc = BatchNormalization(axis=axis)(sc)
    c50 = Activation('relu')(c50)
    c51 = Activation('relu')(c51)
    c52 = Activation('relu')(c52)
    p5 = Activation('relu')(p5)
    sc = Activation('relu')(sc)
    r5 = concatenate([c51,c50,c52,p5,sc])
    r5 = Conv2D(8*KERNELS, (1, 1), activation='relu', padding='same') (r5)
    
    ## Expansion Phase ##
    #uconv_16 = Conv2DTranspose(8*KERNELS, (2, 2), strides=(2, 2), padding='same') (r5)
    #uconv_16 = concatenate([uconv_16, conv_16])
    #uconv_16 = inception_conv_layer(8*KERNELS, dropout_rate, batch_norm, residual)(uconv_16)
    
    uconv_32 = Conv2DTranspose(4*KERNELS, (2, 2), strides=(2, 2), padding='same') (r5) #(uconv_16)
    uconv_32 = concatenate([uconv_32, conv_32])
    uconv_32 = inception_conv_layer(4*KERNELS, dropout_rate, batch_norm, residual)(uconv_32)
    
    uconv_64 = Conv2DTranspose(2*KERNELS, (2, 2), strides=(2, 2), padding='same') (uconv_32)
    uconv_64 = concatenate([uconv_64, conv_64])
    uconv_64 = inception_conv_layer(2*KERNELS, dropout_rate, batch_norm, residual)(uconv_64)
    
    uconv_128 = Conv2DTranspose(KERNELS, (2, 2), strides=(2, 2), padding='same') (uconv_64)
    uconv_128 = concatenate([uconv_128, conv_128])
    uconv_128 = inception_conv_layer(KERNELS, dropout_rate, batch_norm, residual)(uconv_128)
    
    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid') (uconv_128)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model