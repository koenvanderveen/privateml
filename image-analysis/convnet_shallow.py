import keras
import numpy as np
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from pond.nn import Dense, Sigmoid, Relu, ReluExact, Reveal, Diff, Softmax, CrossEntropy, Sequential, DataLoader, Conv2D, \
    AveragePooling2D, Flatten
from keras.utils import to_categorical
import datetime

# Read data.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:, np.newaxis, :, :] / 255.0
x_test = x_test[:, np.newaxis, :, :] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# NativeTensor.
classifier = Sequential([
    Conv2D((3, 3, 1, 32), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp),
           l2reg_lambda=5.0),
    ReluExact(),
    AveragePooling2D(pool_size=(4, 4)),
    Flatten(),
    Dense(10, 1568, l2reg_lambda=5.0),
    Reveal(),
    Softmax()
])


classifier.initialize()
classifier.fit(
    x_train=DataLoader(x_train, wrapper=NativeTensor),
    y_train=DataLoader(y_train, wrapper=NativeTensor),
    x_valid=DataLoader(x_test, wrapper=NativeTensor),
    y_valid=DataLoader(y_test, wrapper=NativeTensor),
    loss=CrossEntropy(),
    epochs=5,
    batch_size=64,
    verbose=1,
)

exit()

# # PublicEncodedTensor
# classifier.initialize()
# classifier.fit(
#     x_train=DataLoader(x_train, wrapper=PublicEncodedTensor),
#     y_train=DataLoader(y_train, wrapper=PublicEncodedTensor),
#     x_valid=DataLoader(x_test, wrapper=PublicEncodedTensor),
#     y_valid=DataLoader(y_test, wrapper=PublicEncodedTensor),
#     loss=CrossEntropy(),
#     epochs=1,
#     batch_size=32,
#     verbose=1,
# )
#
# # PrivateEncodedTensor
# classifier.initialize()
# start = datetime.now()
# classifier.fit(
#     x_train=DataLoader(x_train, wrapper=PrivateEncodedTensor),
#     y_train=DataLoader(y_train, wrapper=PrivateEncodedTensor),
#     loss=CrossEntropy(),
#     epochs=1,
#     batch_size=32,
#     verbose=1
# )

