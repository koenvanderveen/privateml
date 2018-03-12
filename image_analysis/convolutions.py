import keras
import numpy as np
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from pond.nn import Dense, Conv2D
import time


# read data
image_batch = np.arange(3*5*6*6).reshape(3, 5, 6, 6)

print(image_batch)


# forward pass
conv_layer = Conv2D((4, 4, 5, 5), strides=2, filter_init=lambda shp: np.arange(np.prod(shp)).reshape(shp))
conv_layer.initialize()
start = time.time()
output = conv_layer.forward(NativeTensor(image_batch))
print(output)
#backward pass (with random update)
delta = NativeTensor(np.random.normal(size=output.shape))
lr = 0.01
_ = conv_layer.backward(d_y=delta, learning_rate=lr)
print(time.time()-start)