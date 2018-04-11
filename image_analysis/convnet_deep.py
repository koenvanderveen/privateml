import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import numpy as np
    import keras
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from pond.nn import Dense, Relu, Reveal, CrossEntropy, SoftmaxStable, Sequential, DataLoader, Conv2D, AveragePooling2D, Flatten
from keras.utils import to_categorical
np.random.seed(40)

_ = np.seterr(over='raise')
_ = np.seterr(under='raise')
_ = np.seterr(invalid='raise')


# read data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:,np.newaxis,:,:] / 255.0
x_test = x_test[:,np.newaxis,:,:] / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


tensortype = NativeTensor
batch_size = 128
input_shape = [batch_size] + list(x_train.shape[1:])

convnet_deep = Sequential([
    Conv2D((3, 3, 1, 32), strides=1, padding=1, filter_init=lambda shp: np.random.uniform(low=-0.14, high=0.14, size=shp)),
    AveragePooling2D(pool_size=(2, 2)),
    Relu(),
    Conv2D((3, 3, 32, 32), strides=1, padding=1, filter_init=lambda shp: np.random.uniform(low=-0.1, high=0.1, size=shp)),
    AveragePooling2D(pool_size=(2, 2)),
    Relu(),
    Flatten(),
    Dense(10, 1568),
    Reveal(),
    SoftmaxStable()
])


convnet_deep.initialize(initializer=tensortype, input_shape=input_shape)
convnet_deep.fit(
    x_train=DataLoader(x_train, wrapper=tensortype),
    y_train=DataLoader(y_train, wrapper=tensortype),
    x_valid=DataLoader(x_test, wrapper=tensortype),
    y_valid=DataLoader(y_test, wrapper=tensortype),
    loss=CrossEntropy(),
    epochs=5,
    batch_size=batch_size,
    verbose=1,
    learning_rate=0.01
)

