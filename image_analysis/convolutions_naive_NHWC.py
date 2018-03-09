import numpy as np
import keras
import time
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
        s = (x.shape[1] - self.fshape[0]) // self.strides + 1
        self.initializer = type(x)
        fmap = self.initializer(np.zeros((x.shape[0], s, s, self.fshape[-1])))
        for j in range(s):
            for i in range(s):
                fmap[:, j, i, :] = (x[:, j * self.strides:j * self.strides + self.fshape[0],
                                    i * self.strides:i * self.strides + self.fshape[1], :,
                                    np.newaxis] * self.filters).sum(axis=(1, 2, 3))
        self.cache = x
        return fmap

    def backward(self, d_y, learning_rate):
        x = self.cache
        # compute gradients for internal parameters and update
        d_weights = self.get_grad(x, d_y)
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

    def get_grad(self, x, layer_err):
        total_layer_err = layer_err.sum(axis=(0, 1, 2))
        filters_err = self.initializer(np.zeros(self.fshape))
        s = (x.shape[1] - self.fshape[0]) // self.strides + 1
        summed_x = x.sum(axis=0)
        for j in range(s):
            for i in range(s):
                filters_err += summed_x[j * self.strides:j * self.strides + self.fshape[0],
                               i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis]
        return filters_err * total_layer_err



# read data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
image_batch = x_train[0:64, :, :, np.newaxis] / 255.0


conv_layer = Conv2D((4, 4, 1, 20),
                    strides=2,
                    filter_init=lambda shp: np.random.normal(scale=0.01, size=shp))

conv_layer.initialize()
start = time.time()
output = conv_layer.forward(NativeTensor(image_batch))
print(time.time()-start)
delta = NativeTensor(np.random.normal(size=output.shape))
lr = 0.01
_ = conv_layer.backward(d_y=delta, learning_rate=lr)
print(time.time()-start)


