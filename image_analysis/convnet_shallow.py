import keras
import numpy as np
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from pond.nn import Dense, ReluExact, Relu, Reveal, CrossEntropy, SoftmaxStable,\
                    Sequential, DataLoader, Conv2D, AveragePooling2D, Flatten
from keras.utils import to_categorical
np.random.seed(42)
import datetime

# Read data.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:, np.newaxis, :, :] / 255.0
x_test = x_test[:, np.newaxis, :, :] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

_ = np.seterr(over='raise')
_ = np.seterr(under='raise')
_ = np.seterr(invalid='raise')

tensortype = PrivateEncodedTensor

convnet_shallow = Sequential([
    Conv2D((3, 3, 1, 16), strides=1, padding=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)),
    AveragePooling2D(pool_size=(2, 2)),
    Relu(order=3),
    # ReluExact(),
    Flatten(),
    Dense(10, 3136),
    Reveal(),
    SoftmaxStable()
])

convnet_shallow.initialize()
convnet_shallow.fit(
    x_train=DataLoader(x_train, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_test, wrapper=tensortype),
    y_valid=DataLoader(y_test, wrapper=tensortype),
    loss=CrossEntropy(),
    epochs=5,
    batch_size=128,
    verbose=1,
    learning_rate=0.01
)

