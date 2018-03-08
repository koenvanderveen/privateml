import numpy as np
import sys
from datetime import datetime
from functools import reduce
from pond.tensor import NativeTensor, PrivateEncodedTensor, PublicEncodedTensor
from im2col import im2col_indices, col2im_indices
import math


class Layer:
    pass


class Dense(Layer):

    def __init__(self, num_nodes, num_features, initial_scale=.01, l2reg_lambda=0.0):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.initial_scale = initial_scale
        self.l2reg_lambda = l2reg_lambda
        self.weights = None
        self.bias = None

    def initialize(self):
        self.weights = NativeTensor(np.random.randn(self.num_features, self.num_nodes) * self.initial_scale)
        self.bias = NativeTensor(np.zeros((1, self.num_nodes)))

    def forward(self, x):
        y = x.dot(self.weights) + self.bias
        # cache result for backward pass
        self.cache = x
        return y

    def backward(self, d_y, learning_rate):
        x = self.cache
        # compute gradients for internal parameters and update
        d_weights = x.transpose().dot(d_y)
        d_weights += self.weights * (self.l2reg_lambda / x.shape[0])

        d_bias = d_y.sum(axis=0)
        self.weights = (d_weights * learning_rate).neg() + self.weights
        self.bias = (d_bias * learning_rate).neg() + self.bias
        # compute and return external gradient
        d_x = d_y.dot(self.weights.transpose())
        return d_x


class SigmoidExact(Layer):

    def __init__(self):
        self.cache = None

    def initialize(self):
        pass

    def forward(self, x):
        y = (x.neg().exp() + 1).inv()
        self.cache = y
        return y

    def backward(self, d_y, learning_rate):
        y = self.cache
        d_x = d_y * y * (y.neg() + 1)
        return d_x


class Sigmoid(Layer):

    def __init__(self):
        self.cache = None

    def initialize(self):
        pass

    def forward(self, x):
        w0 = 0.5
        w1 = 0.2159198015
        w3 = -0.0082176259
        w5 = 0.0001825597
        w7 = -0.0000018848
        w9 = 0.0000000072

        x2 = x * x
        x3 = x2 * x
        x5 = x2 * x3
        x7 = x2 * x5
        x9 = x2 * x7
        y = x9*w9 + x7*w7 + x5*w5 + x3*w3 + x*w1 + w0

        self.cache = y
        return y

    def backward(self, d_y, learning_rate):
        y = self.cache
        d_x = d_y * y * (y.neg() + 1)
        return d_x


class Softmax(Layer):

    def __init__(self):
        pass

    def initialize(self):
        pass

    def forward(self, x):
        # we add the - x.max() for numerical stability, i.e. to prevent overflow
        likelihoods = (x - x.max(axis=1, keepdims=True)).exp()
        probs = likelihoods.div(likelihoods.sum(axis=1, keepdims=True))
        self.cache = probs
        return probs

    def backward(self, d_probs, learning_rate):
        # TODO does the split between Softmax and CrossEntropy make sense?
        probs = self.cache
        batch_size = probs.shape[0]
        d_scores = probs - d_probs
        d_scores = d_scores.div(batch_size)
        return d_scores


class ReluExact(Layer):

    def __init__(self):
        self.cache = None

    def initialize(self):
        pass

    def forward(self, x):
        y = x * (x > 0)
        self.cache = x
        return y

    def backward(self, d_y, learning_rate):
        x = self.cache
        d_x = (x > 0) * d_y
        return d_x


class Relu(Layer):

    def __init__(self, order=9, domain=(-100, 100)):
        self.cache = None
        self.n_coeff = order + 1
        self.coeff = NativeTensor(self.compute_coefficients_relu(order, domain))
        self.coeff_der = (self.coeff * NativeTensor(list(range(self.n_coeff))[::-1]))[:-1]

    def initialize(self):
        pass

    def forward(self, x):
        x.expand_dims(axis=4).repeat(self.n_coeff, axis=4)

        for i in range(self.n_coeff)[::-1]:
            # x[:, :, :, :, i] = x[:, :, :, :, i] ** i
            x[:, :, :, :, i] **= i

        y = x.dot(self.coeff)
        self.cache = x[:, :, :, :, 1:]
        return y

    def backward(self, d_y, learning_rate):
        x = self.cache
        d_x = d_y * x.dot(self.coeff_der)
        return d_x

    def relu_der(self, x):
        x_powers = np.array([x ** i for i in range(self.n_coeff - 1)][::-1]).T
        y = x_powers.dot(self.coeff_der)
        return y

    @staticmethod
    def compute_coefficients_relu(order, domain):
        assert domain[0] < 0 < domain[1]
        x = list(range(domain[0], domain[1]))
        y = [0] * abs(domain[0]) + list(range(0, domain[1]))
        return np.polyfit(x, y, order)


class Dropout(Layer):
    # TODO
    pass


class Flatten(Layer):
    def __init__(self):
        self.shape = None

    def initialize(self):
        pass

    def forward(self, x):
        self.shape = x.shape
        y = x.reshape(x.shape[0], -1)
        return y

    def backward(self, d_y, learning_rate):
        return d_y.reshape(self.shape)


class Conv2D():
    def __init__(self, fshape, strides=1, padding=0, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp),
                 l2reg_lambda=0.0, channels_first=True):
        """ 2 Dimensional convolutional layer, expects NCHW data format
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
        self.l2reg_lambda = l2reg_lambda
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

        X_col = im2col_indices(x, field_height=h_filter, field_width=w_filter, padding=self.padding,
                               stride=self.strides)
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
        dw += self.filters * (self.l2reg_lambda / self.cached_input_shape[0])

        self.filters = (dw * learning_rate).neg() + self.filters

        W_reshape = self.filters.reshape(n_filter, -1)
        dx_col = W_reshape.transpose().dot(dout_reshaped)
        dx = col2im_indices(dx_col, self.cached_input_shape, self.initializer,field_height=h_filter,
                            field_width=w_filter, padding=self.padding, stride=self.strides)

        return dx


class Conv2DNaive():

    def __init__(self, fshape, strides=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)):
        """ 2 Dimensional convolutional layer NHWC
            fshape: tuple of rank 4
            strides: int with stride size
            filter init: lambda function with shape parameter
            Example: Conv2D((4, 4, 1, 20), strides=2, filter_init=lambda shp: np.random.normal(scale=0.01, size=shp))
        """
        self.fshape = fshape
        self.strides = strides
        self.filter_init = filter_init
        self.cache = None
        self.initializer = None

    def initialize(self):
        self.filters = self.filter_init(self.fshape)

    def forward(self, x):
        # TODO: padding
        s = (x.shape[1] - self.fshape[0]) // self.strides + 1
        self.initializer = type(x)
        fmap = self.initializer(np.zeros((x.shape[0], s, s, self.fshape[-1])))
        for j in range(s):
            for i in range(s):
                fmap[:, j, i, :] = (x[:, j * self.strides:j * self.strides + self.fshape[0],
                                    i * self.strides:i * self.strides + self.fshape[1],
                                    :, np.newaxis] * self.filters).sum(axis=(1, 2, 3))
        self.cache = x
        return fmap

    def backward(self, d_y, learning_rate):
        x = self.cache
        # compute gradients for internal parameters and update
        d_weights = self.get_grad(x, d_y)
        self.filters = (d_weights * learning_rate).neg() + self.filters
        # compute and return external gradient
        d_x = self.backwarded_error(d_y)
        return d_x

    def backwarded_error(self, layer_err):
        bfmap_shape = (layer_err.shape[1] - 1) * self.strides + self.fshape[0]
        backwarded_fmap = self.initializer(np.zeros((layer_err.shape[0], bfmap_shape, bfmap_shape, self.fshape[-2])))
        s = (backwarded_fmap.shape[1] - self.fshape[0]) // self.strides + 1
        for j in range(s):
            for i in range(s):
                backwarded_fmap[:, j * self.strides:j  * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1]] += (self.filters[np.newaxis, ...] * layer_err[:, j:j+1, i:i+1, np.newaxis, :]).sum(axis=4)
        return backwarded_fmap

    def get_grad(self, x, layer_err):
        total_layer_err = layer_err.sum(axis=(0, 1, 2))
        filters_err = self.initializer(np.zeros(self.fshape))
        s = (x.shape[1] - self.fshape[0]) // self.strides + 1
        summed_x = x.sum(axis=0)
        for j in range(s):
            for i in range(s):
                filters_err += summed_x[j  * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis]
        return filters_err * total_layer_err


class AveragePooling2D():

    def __init__(self, pool_size, strides=None, channels_first=True):
        """ Average Pooling layer NCHW
            pool_size: (n x m) tuple
            strides: int with stride size
            Example: AveragePooling2D(pool_size=(2,2))
        """
        self.pool_size = pool_size
        self.pool_area = pool_size[0] * pool_size[1]
        self.cache = None
        self.initializer = None
        if strides == None:
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
        d_x = (d_y_expanded * x) / self.pool_area
        return d_x


class AveragePooling2D_NHWC():
    def __init__(self, pool_size, strides=None):
        """ Average Pooling layer
            pool_size: (n x m) tuple
            strides: int with stride size
            Example: AveragePooling2D(pool_size=(2,2))
        """
        self.pool_size = pool_size
        self.pool_area = pool_size[0] * pool_size[1]
        self.cache = None
        self.initializer = None
        if strides == None:
            self.strides = pool_size[0]
        else:
            self.strides = strides

    def initialize(self):
        pass

    def forward(self, x):
        s = (x.shape[1] - self.pool_size[0]) // self.strides + 1
        self.initializer = type(x)
        pooled = self.initializer(np.zeros((x.shape[0], s, s, x.shape[3])))
        for j in range(s):
            for i in range(s):
                pooled[:, j, i, :] = x[:, j * self.strides:j * self.strides + self.pool_size[0],
                                     i * self.strides:i * self.strides + self.pool_size[1], :].sum(axis=(1, 2))

        pooled = pooled / self.pool_area
        self.cache = x
        return pooled

    def backward(self, d_y, learning_rate):
        x = self.cache
        d_y_expanded = d_y.repeat(self.pool_size[0], axis=1)
        d_y_expanded = d_y_expanded.repeat(self.pool_size[1], axis=2)
        d_x = d_y_expanded * x / self.pool_area
        return d_x


class Reveal(Layer):

    def __init__(self):
        pass

    def initialize(self):
        pass

    def forward(self, x):
        return x.reveal()

    def backward(self, d_y, learning_rate):
        return d_y


class Loss:
    pass


class Diff(Loss):

    def derive(self, y_pred, y_train):
        return y_pred - y_train


class CrossEntropy(Loss):

    def evaluate(self, probs_pred, probs_correct):
        batch_size = probs_pred.shape[0]
        losses = (probs_correct * probs_pred.log()).neg().sum(axis=1)
        loss = losses.sum(axis=0).div(batch_size)
        return loss

    def derive(self, y_pred, y_correct):
        return y_correct


class SoftmaxCrossEntropy(Loss):
    # def evaluate(self, probs_pred, probs_correct):
    #     batch_size = probs_pred.shape[0]
    #     losses = (probs_correct * (probs_pred).log()).neg()
    #     loss = losses.sum(axis=0).div(batch_size)
    #     return loss
    #
    # def derive(self, y_pred, y_correct):
    #     return y_correct
    pass


class DataLoader:

    def __init__(self, data, wrapper=lambda x: x):
        self.data = data
        self.wrapper = wrapper

    def batches(self, batch_size=None, shuffle_indices=None):
        if shuffle_indices is not None:
            self.data = self.data[shuffle_indices]
        if batch_size is None:
            batch_size = self.data.shape[0]
        return (
            self.wrapper(self.data[i:i+batch_size])
            for i in range(0, self.data.shape[0], batch_size)
        )

    def all_data(self):
        return self.wrapper(self.data)


class Model:
    pass


class Sequential(Model):

    def __init__(self, layers):
        self.layers = layers

    def initialize(self):
        for layer in self.layers:
            layer.initialize()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_y, learning_rate):
        max_dy = 1000.0
        for layer in reversed(self.layers):
            # d_y = layer.backward(d_y, learning_rate).clip(-max_dy, max_dy)
            d_y = layer.backward(d_y, learning_rate)

    @staticmethod
    def print_progress(batch_index, n_batches, batch_size, train_loss=None, train_acc=None,
                       val_loss = None, val_acc=None):
        sys.stdout.write('\r')
        sys.stdout.flush()
        progress = (batch_index / n_batches)
        n_eq = int(progress * 30)
        n_dot = 60 - n_eq
        progress_bar = "=" * n_eq + ">" + n_dot * "."

        if val_loss is None:
            message = "{}/{} [{}] - train_loss: {:.5f} - train_acc {:.5f}"
            sys.stdout.write(message.format((batch_index+1) * batch_size, n_batches * batch_size, progress_bar,
                                            train_loss, train_acc))
        else:
            message = "{}/{} [{}] - train_loss: {:.5f} - train_acc {:.5f} - val_loss {:.5f} - val_acc {:.5f}"
            sys.stdout.write(message.format((batch_index+1) * batch_size, n_batches * batch_size, progress_bar, train_loss,
                                            train_acc, val_loss, val_acc))

        sys.stdout.flush()

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, loss=None, batch_size=32, epochs=1000,
            learning_rate=.01, verbose=0):

        if not isinstance(x_train, DataLoader): x_train = DataLoader(x_train)
        if not isinstance(y_train, DataLoader): y_train = DataLoader(y_train)

        if x_valid is not None:
            if not isinstance(x_valid, DataLoader): x_valid = DataLoader(x_valid)
            if not isinstance(y_train, DataLoader): y_valid = DataLoader(y_valid)

        for epoch in range(epochs):
            if verbose >= 1:
                print(datetime.now(), "Epoch %s" % epoch)

            # Create batches on shuffled data
            shuffle = np.random.permutation(x_train.data.shape[0])
            batches = zip(x_train.batches(batch_size, shuffle_indices=shuffle),
                          y_train.batches(batch_size, shuffle_indices=shuffle))

            n_batches = math.ceil(len(x_train.data) / batch_size)
            for batch_index, (x_batch, y_batch) in enumerate(batches):
                if verbose >= 2:
                    print(datetime.now(), "Batch %s" % batch_index)

                y_pred = self.forward(x_batch)
                train_loss = loss.evaluate(y_pred, y_batch).unwrap()[0]
                acc = np.mean(y_batch.unwrap().argmax(axis=1) == y_pred.unwrap().argmax(axis=1))
                d_y = loss.derive(y_pred, y_batch)
                self.backward(d_y, learning_rate)

                # print status
                if verbose >= 1:
                    if batch_index % 10 == 0:
                        # validation print
                        y_pred_val = self.predict(x_valid)
                        val_loss = np.sum(loss.evaluate(y_pred_val, y_valid.all_data()).unwrap())
                        val_acc = np.mean(y_valid.all_data().unwrap().argmax(axis=1) == y_pred_val.unwrap().argmax(axis=1))
                        self.print_progress(batch_index, n_batches, batch_size, train_acc=acc, train_loss=train_loss,
                                            val_loss=val_loss, val_acc=val_acc)
                    # elif batch_index + 1 != n_batches:
                    else:
                        # normal print
                        self.print_progress(batch_index, n_batches, batch_size, train_acc=acc, train_loss=train_loss)

            # Newline after progressbar.
            print()

    def predict(self, x, batch_size=32, verbose=0):
        if not isinstance(x, DataLoader): x = DataLoader(x)
        batches = []
        for batch_index, x_batch in enumerate(x.batches(batch_size)):
            if verbose >= 2: print(datetime.now(), "Batch %s" % batch_index)
            y_batch = self.forward(x_batch)
            batches.append(y_batch)
        return reduce(lambda x, y: x.concatenate(y), batches)
