import keras
import numpy as np
from pond.tensor import PublicEncodedTensor, decode, encode

np.random.seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train[:, np.newaxis, :, :] / 255.0
x = PublicEncodedTensor(x_train[0:128])

X_col = x.im2col(3, 3, 1, 1)
w = PublicEncodedTensor(np.random.normal(scale=0.1, size=(16,9)))


print(w.dot(X_col).unwrap().max())






