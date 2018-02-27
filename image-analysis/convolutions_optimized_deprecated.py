import numpy as np
import keras
import time
import math
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor


class Conv2D():
    def __init__(self, fshape, strides=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)):
        """ 2 Dimensional convolutional layer
            fshape: tuple of rank 4
            strides: int with stride size
            filter init: lambda function with shape parameter
            Example: Conv2D((4, 4, 1, 20), strides=2, filter_init=lambda shp: np.random.normal(scale=0.01,
            size=shp))
        """
        self.fshape = fshape
        self.strides = strides
        self.filter_init = filter_init
        self.cache = None
        self.initializer = None

    def initialize(self):
        self.filters = self.filter_init(self.fshape)

    def forward(self, x):
        # TODO: padding
        self.initializer = type(x)
        x_patches_flat = im2col_indices(x, field_height=self.fshape[0], field_width=self.fshape[1]).transpose()

        print(x_patches_flat.shape)
        filters_flat = np.reshape(self.filters, [-1, self.filters.shape[-1]])
        fmaph = fmapw = int(math.sqrt(x_patches_flat.shape[0] / self.fshape[3]))
        print(fmaph)
        fmap = x_patches_flat.dot(filters_flat)
        fmap = fmap.reshape([x.shape[0], fmaph, fmapw, self.fshape[-1]])
        self.cache = x_patches_flat
        return fmap

    def backward(self, d_y, learning_rate):
        x_patches_flat = self.cache
        # compute gradients for internal parameters and update
        d_weights = self.get_grad(x_patches_flat, d_y)
        self.filters = (d_weights * learning_rate).neg() + self.filters
        # compute and return external gradient
        d_x = self.backwarded_error(d_y)
        return d_x

    def backwarded_error(self, layer_err):
        bfmap_shape = (layer_err.shape[1] - 1) * self.strides + self.fshape[0]
        backwarded_fmap = self.initializer(np.zeros((layer_err.shape[0], bfmap_shape, bfmap_shape, self.fshape[-2])))
        s = (backwarded_fmap.shape[1] - self.fshape[0]) // self.strides + 1
        for j in range(s):
            for i in range(s):
                backwarded_fmap[:, j * self.strides:j * self.strides + self.fshape[0],
                i * self.strides:i * self.strides + self.fshape[1]] += (
                self.filters[np.newaxis, ...] * layer_err[:, j:j + 1, i:i + 1, np.newaxis, :]).sum(axis=4)
        return backwarded_fmap

    def get_grad(self, x_patches_flat, layer_err):
        dout_reshaped = layer_err.transpose(3, 1, 2, 0).reshape((self.fshape[-1], -1))
        print(x_patches_flat.shape)

        d_weights = dout_reshaped.dot(x_patches_flat.transpose())
        d_weights = d_weights.reshape(self.fshape)
        return d_weights


        # total_layer_err = layer_err.sum(axis=(0, 1, 2))
        # # (20,)
        #
        # filters_err = self.initializer(np.zeros(self.fshape))
        # # (4,4,1,20)
        #
        # s = (x.shape[1] - self.fshape[0]) // self.strides + 1
        # # 13
        #
        # summed_x = x.sum(axis=0)
        # # (28,1,1)
        # for j in range(s):
        #     for i in range(s):
        #         filters_err += summed_x[j * self.strides:j * self.strides + self.fshape[0],
        #                        i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis]
        # return filters_err * total_layer_err


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, H, W, C = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (i, j, k)


def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = x.pad(((0, 0), (p, p), (p, p), (0, 0)), mode='constant')
    i, j, k = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)
    cols = x_padded[:, i, j, k]
    print(cols.shape)
    C = x.shape[3]
    cols = cols.transpose(1, 2, 0).reshape((field_height * field_width * C, -1))
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# read data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
image_batch = x_train[0:64, :, :, np.newaxis] / 255.0


conv_layer = Conv2D((4, 4, 1, 20), strides=2, filter_init=lambda shp: np.random.normal(scale=0.01, size=shp))
conv_layer.initialize()
start = time.time()
output = conv_layer.forward(NativeTensor(image_batch))
print(time.time()-start)
print(output.shape)
delta = NativeTensor(np.random.normal(size=output.shape))
lr = 0.01
_ = conv_layer.backward(d_y=delta, learning_rate=lr)
print(time.time()-start)









