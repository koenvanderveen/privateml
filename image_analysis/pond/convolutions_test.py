import keras
import numpy as np
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from pond.nn import Dense, Conv2D, Conv2DNaive
import time


# read data
image_batch = np.arange(1*2*3*3).reshape(1, 2, 3, 3).astype('float')


# forward pass
conv_layer = Conv2D((2, 2, 2, 2), strides=1, filter_init=lambda shp: np.arange(np.prod(shp)).reshape(shp).astype('float'))
conv_layer.initialize()
output = conv_layer.forward(NativeTensor(image_batch))



image_batch2 = image_batch.transpose(0,2,3,1)
conv_layer2 = Conv2DNaive((2, 2, 2, 2), strides=1, filter_init=lambda shp: np.arange(np.prod(shp)).reshape(shp).astype('float'))
conv_layer2.initialize()
output2 = conv_layer2.forward(NativeTensor(image_batch2))


output2_reshaped = output2.transpose(0,3,1,2)

print((image_batch[:,:,:2,:2] * conv_layer.filters[:,:,:,0].unwrap().transpose(2,0,1)).sum())

print(output[0,0,0,0])
print(output2_reshaped[0,0,0,0])
#backward pass (with random update)

# delta = NativeTensor(np.random.normal(size=output.shape))
# lr = 0.01
# _ = conv_layer.backward(d_y=delta, learning_rate=lr)
# print(time.time()-start)