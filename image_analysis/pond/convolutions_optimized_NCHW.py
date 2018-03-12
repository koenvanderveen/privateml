import numpy as np
import keras
import time
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from im2col import im2col_indices, col2im_indices

np.random.seed(0)

class Conv2D():
    def __init__(self, fshape, strides=1, padding=0, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp),
                 channels_first=True):
        """ 2 Dimensional convolutional layer
            fshape: tuple of rank 4
            strides: int with stride size
            filter init: lambda function with shape parameter
            Example: Conv2D((4, 4, 1, 20), strides=2, filter_init=lambda shp: np.random.normal(scale=0.01,
            size=shp))
        """
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.filter_init = filter_init
        self.cache = None
        self.cached_input_shape = None
        self.initializer = None
        assert channels_first

    def initialize(self):
        self.filters = NativeTensor(self.filter_init(self.fshape))

    def forward(self, x):
        # TODO: padding
        self.initializer = type(x)

        # shapes, assuming NCHW
        h_filter, w_filter, d_filters, n_filters = self.filters.shape
        n_x, d_x, h_x, w_x = x.shape


        h_out = int((h_x - h_filter + 2 * self.padding) / self.strides + 1)
        w_out = int((w_x - w_filter + 2 * self.padding) / self.strides + 1)

        X_col = im2col_indices(x, field_height=h_filter, field_width=w_filter,
                               padding=self.padding, stride=self.strides)
        W_col = self.filters.reshape(n_filters, -1)
        # multiplication
        out = W_col.dot(X_col)
        out = out.reshape(self.fshape[3], h_out, w_out, n_x)

        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        self.cache = X_col
        self.cached_input_shape = x.shape

        return out

    def backward(self, d_y, learning_rate):
        X_col = self.cache
        h_filter, w_filter, d_filter, n_filter = self.filters.shape

        dout_reshaped = d_y.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dw = dout_reshaped.dot(X_col.transpose())
        dw = dw.reshape(self.filters.shape)
        self.filters = (dw * learning_rate).neg() + self.filters

        W_reshape = self.filters.reshape(n_filter, -1)
        dx_col = W_reshape.transpose().dot(dout_reshaped)
        dx = col2im_indices(dx_col, self.cached_input_shape, self.initializer,field_height=h_filter,
                            field_width=w_filter, padding=self.padding, stride=self.strides)

        return dx



# read data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
image_batch = x_train[0:64, np.newaxis, :, :] / 255.0


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









