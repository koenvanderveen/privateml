import keras
import numpy as np
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from pond.nn import Dense, Relu, Reveal, CrossEntropy, SoftmaxStable, Sequential, DataLoader, Conv2D, \
    AveragePooling2D, Flatten
from keras.utils import to_categorical

# Read data.
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:, np.newaxis, :, :] / 255.0
x_test = x_test[:, np.newaxis, :, :] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

x_train = x_train[:128]
x_test = x_test[:128]
y_train = y_train[:128]
y_test = y_train[:128]

_ = np.seterr(over='raise')
_ = np.seterr(under='raise')
_ = np.seterr(invalid='raise')

batch_size = 128
input_shape = [batch_size] + list(x_train.shape[1:])

for tensortype in [NativeTensor, PublicEncodedTensor, PrivateEncodedTensor]:
    np.random.seed(42)
    convnet_shallow = Sequential([
        Conv2D((3, 3, 1, 32), strides=1, padding=1),
        AveragePooling2D(pool_size=(2, 2)),
        Relu(order=3),
        Flatten(),
        Dense(10, 6272),
        Reveal(),
        SoftmaxStable()
    ])

    convnet_shallow.initialize(input_shape=input_shape, initializer=tensortype)
    convnet_shallow.fit(
        x_train=DataLoader(x_train, wrapper=tensortype),
        y_train=DataLoader(y_train, wrapper=tensortype),
        x_valid=DataLoader(x_test, wrapper=tensortype),
        y_valid=DataLoader(y_test, wrapper=tensortype),
        loss=CrossEntropy(),
        epochs=1,
        batch_size=batch_size,
        verbose=1,
        learning_rate=0.01
    )

