import numpy as np
import keras
import time
import math
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor


class AveragePooling2D():

    def __init__(self, pool_size, strides=None, channels_first=True):
        """ Average Pooling layer
            pool_size: (n x m) tuple
            strides: int with stride size
            Example: AveragePooling2D(pool_size=(2,2))
        """
        self.pool_size = pool_size
        self.pool_area = pool_size[0] * pool_size[1]
        self.cache = None
        self.initializer = None
        if strides is None:
            self.strides = pool_size[0]
        else:
            self.strides = strides

        assert channels_first

    def initialize(self):
        pass

    def forward(self,x):
        # forward pass of average pooling, assumes NCHW data format

        s = (x.shape[2] - self.pool_size[0]) // self.strides + 1
        self.initializer = type(x)
        pooled = self.initializer(np.zeros((x.shape[0], x.shape[1], s, s)))
        for j in range(s):
            for i in range(s):
                pooled[:, :, j, i] = x[:, :, j * self.strides:j * self.strides + self.pool_size[0],
                                       i * self.strides:i * self.strides + self.pool_size[1]].sum(axis=(2, 3))

        pooled = pooled / self.pool_area
        self.cache = x
        return pooled

    def backward(self, d_y, learning_rate):
        x = self.cache
        d_y_expanded = d_y.repeat(self.pool_size[0], axis=2)
        d_y_expanded = d_y_expanded.repeat(self.pool_size[1], axis=3)
        d_x = d_y_expanded * x / self.pool_area
        return d_x



# read data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
image_batch = x_train[0:64, np.newaxis, :, :] / 255.0


# forward pass
start = time.time()
avg_pooling_layer = AveragePooling2D(pool_size=(2,2))
output = avg_pooling_layer.forward(NativeTensor(image_batch))
print(time.time()-start)
print(output.shape)
# backward pass (with random update)
delta = NativeTensor(np.random.normal(loc=1, scale=100, size=output.shape))
lr = 0.01
_ = avg_pooling_layer.backward(d_y=delta, learning_rate=lr)
print(time.time()-start)









