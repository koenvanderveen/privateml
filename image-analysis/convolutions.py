import numpy as np
import keras
import time
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from pond.nn import Conv2D

# read data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
image_batch = x_train[0:64,:,:,np.newaxis] / 255.0


conv_layer = Conv2D((4, 4, 1, 20), strides=2, filter_init=lambda shp: np.random.normal(scale=0.01, size=shp))
conv_layer.initialize()
start = time.time()
output = conv_layer.forward(PrivateEncodedTensor(image_batch))
print(time.time()-start)
