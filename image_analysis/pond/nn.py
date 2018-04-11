import numpy as np
import sys
from datetime import datetime, timedelta
from functools import reduce
from pond.tensor import NativeTensor, PublicEncodedTensor, PrivateEncodedTensor, stack
import pond.tensor as t
import math
import time



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
        self.cache = None

    def initialize(self, input_shape, initializer=None, model=None):
        if initializer is not None:
            self.weights = initializer(np.random.randn(self.num_features, self.num_nodes) * self.initial_scale)
            self.bias = initializer(np.zeros((1, self.num_nodes)))
        output_shape = [input_shape[0]] + [self.num_nodes]
        return output_shape

    def forward(self, x):
        y = x.dot(self.weights) + self.bias
        self.cache = x
        return y

    def backward(self, d_y, learning_rate):
        x = self.cache
        d_x = d_y.dot(self.weights.transpose())
        d_weights = x.transpose().dot(d_y)
        if self.l2reg_lambda > 0:
            d_weights = d_weights + self.weights * (self.l2reg_lambda / x.shape[0])

        d_bias = d_y.sum(axis=0)
        # update weights and bias
        self.weights = (d_weights * learning_rate).neg() + self.weights
        self.bias = (d_bias * learning_rate).neg() + self.bias

        return d_x


class SigmoidExact(Layer):

    def __init__(self):
        self.cache = None

    def initialize(self, input_shape, **kwargs):
        return input_shape

    def forward(self, x):
        y = (x.neg().exp() + 1).inv()
        self.cache = y
        return y

    def backward(self, d_y, *_):
        y = self.cache
        d_x = d_y * y * (y.neg() + 1)
        return d_x


class Sigmoid(Layer):

    def __init__(self):
        self.cache = None

    def initialize(self, input_shape, **kwargs):
        return input_shape

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


class SoftmaxStable(Layer):

    def __init__(self):
        pass

    def initialize(self, input_shape, **kwargs):
        self.cache = None
        return input_shape

    def forward(self, x):
        # we add the - x.max() for numerical stability, i.e. to prevent overflow
        likelihoods = (x - x.max(axis=1, keepdims=True)).clip(-10.0, np.inf).exp()
        probs = likelihoods.div(likelihoods.sum(axis=1, keepdims=True))
        self.cache = probs
        return probs

    def backward(self, d_probs, learning_rate):
        probs = self.cache
        batch_size = probs.shape[0]
        d_scores = probs - d_probs
        d_scores = d_scores.div(batch_size)
        return d_scores


class Softmax(Layer):

    def __init__(self):
        pass

    def initialize(self, input_shape, **kwargs):
        return input_shape

    def forward(self, x):
        exp = x.exp()
        probs = exp.div(exp.sum(axis=1, keepdims=True))
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

    def initialize(self, input_shape, **kwargs):
        return input_shape

    def forward(self, x):
        y = x * (x > 0)
        self.cache = x
        return y

    def backward(self, d_y, _):
        x = self.cache
        d_x = (x > 0) * d_y
        return d_x


class Relu(Layer):

    def __init__(self, order=3, domain=(-1, 1), n=1000):
        self.cache = None
        self.n_coeff = order + 1
        self.order = order
        self.coeff = NativeTensor(self.compute_coefficients_relu(order, domain, n))
        self.coeff_der = (self.coeff * NativeTensor(list(range(self.n_coeff))[::-1]))[:-1]
        self.initializer = None
        assert order > 2

    def initialize(self, input_shape, **kwargs):
        return input_shape

    def forward(self, x):
        self.initializer = type(x)

        n_dims = len(x.shape)

        powers = [x, x.square()]
        for i in range(self.order - 2):
            powers.append(x * powers[-1])

        # stack list into tensor
        forward_powers = stack(powers).flip(axis=n_dims)
        y = forward_powers.dot(self.coeff[:-1]) + self.coeff[-1]

        # cache all powers except the last
        self.cache = stack(powers[:-1]).flip(axis=n_dims)
        return y

    def backward(self, d_y, learning_rate):
        powers = self.cache
        c = d_y * self.coeff_der[-1]
        d_y.expand_dims(axis=-1)
        d_x = (d_y * powers).dot(self.coeff_der[:-1]) + c
        return d_x

    @staticmethod
    def compute_coefficients_relu(order, domain, n):
        assert domain[0] < 0 < domain[1]
        x = np.linspace(domain[0], domain[1], n)
        y = (x > 0) * x
        return np.polyfit(x, y, order)


class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate

    def initialize(self, input_shape, **kwargs):
        return input_shape

    def forward(self, x):
        pass

    def backward(self, dx):
        pass


class Flatten(Layer):
    def __init__(self):
        self.shape = None

    def initialize(self, input_shape, **kwargs):
        return [input_shape[0]] + [np.prod(input_shape[1:])]

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
        self.cached_x_col = None
        self.cached_input_shape = None
        self.initializer = None
        assert channels_first

    def initialize(self, input_shape, model=None, initializer=None):
        self.model = model

        if initializer is not None:
            h_filter, w_filter, d_filters, n_filters = self.fshape
            n_x, d_x, h_x, w_x = input_shape
            h_out = int((h_x - h_filter + 2 * self.padding) / self.strides + 1)
            w_out = int((w_x - w_filter + 2 * self.padding) / self.strides + 1)

            self.bias = initializer(np.zeros((n_filters, h_out, w_out)))
            self.filters = initializer(self.filter_init(self.fshape))

        return [n_x, n_filters, h_out, w_out]


    def forward(self, x):
        self.cached_input_shape = x.shape
        self.cache = x
        out, self.cached_x_col = x.conv2d(self.filters, self.strides, self.padding)

        return out + self.bias

    def backward(self, d_y, learning_rate):
        x = self.cache
        h_filter, w_filter, d_filter, n_filter = self.filters.shape
        dx = None

        if self.model.layers.index(self) != 0:
            W_reshaped = self.filters.reshape(n_filter, -1).transpose()
            dout_reshaped = d_y.transpose(1, 2, 3, 0).reshape(n_filter, -1)
            dx = W_reshaped.dot(dout_reshaped).col2im(imshape=self.cached_input_shape, field_height=h_filter,
                                                      field_width=w_filter, padding=self.padding, stride=self.strides)

        d_w = x.conv2d_bw(d_y, self.cached_x_col, self.filters.shape, padding=self.padding, strides=self.strides)
        d_bias = d_y.sum(axis=0)

        if self.l2reg_lambda > 0:
            d_w = d_w + self.filters * (self.l2reg_lambda / self.cached_input_shape[0])

        self.filters = (d_w * learning_rate).neg() + self.filters
        self.bias = (d_bias * learning_rate).neg() + self.bias

        return dx

class ConvAveragePooling2D():
    def __init__(self, fshape, strides=1, padding=0, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp),
                 l2reg_lambda=0.0, pool_size=(2,2), pool_strides=None, channels_first=True):
        """ 2 Dimensional convolutional layer followed by average pooling layer
            , expects NCHW data format and is optimized for communication
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
        self.cache2 = None
        self.cached_input_shape = None
        self.initializer = None

        self.pool_size = pool_size
        self.pool_area = pool_size[0] * pool_size[1]
        if pool_strides == None:
            self.pool_strides = pool_size[0]
        else:
            self.pool_strides = pool_strides

        assert channels_first

    def initialize(self, input_shape, model=None, initializer=None):
        self.model = model

        if initializer is not None:
            h_filter, w_filter, d_filters, n_filters = self.fshape
            n_x, d_x, h_x, w_x = input_shape
            h_out = int((h_x - h_filter + 2 * self.padding) / self.strides + 1)
            w_out = int((w_x - w_filter + 2 * self.padding) / self.strides + 1)

            self.bias = initializer(np.zeros((n_filters, h_out, w_out)))
            self.filters = initializer(self.filter_init(self.fshape))
            s = (h_out - self.pool_size[0]) // self.pool_strides + 1

        return [n_x, n_filters, s, s]


    def forward(self, x):
        self.initializer = type(x)
        self.cached_input_shape = x.shape
        self.cache = x

        out, self.cache2 = x.conv2d(self.filters, self.strides, self.padding)
        x_pool = out + self.bias

        s = (x_pool.shape[2] - self.pool_size[0]) // self.pool_strides + 1
        pooled = self.initializer(np.zeros((x_pool.shape[0], x_pool.shape[1], s, s)))
        for j in range(s):
            for i in range(s):
                pooled[:, :, j, i] = x_pool[:, :, j * self.pool_strides:j * self.pool_strides + self.pool_size[0],
                                            i * self.pool_strides:i * self.pool_strides + self.pool_size[1]].sum(axis=(2, 3))

        pooled = pooled / self.pool_area
        return pooled

    def backward(self, d_y, learning_rate):
        d_y_expanded = d_y.copy().repeat(self.pool_size[0], axis=2)
        d_y_expanded = d_y_expanded.repeat(self.pool_size[1], axis=3)
        d_y_conv = d_y_expanded / self.pool_area
        dx = None
        x = self.cache

        if self.model.layers.index(self) != 0:
            dx = d_y.convavgpool_delta(self.filters, self.cached_input_shape, padding=self.padding,
                                                 strides=self.strides, pool_size=self.pool_size,
                                                 pool_strides=self.pool_strides)

        d_w = x.convavgpool_bw(d_y, self.cache2, self.filters.shape, self.padding, self.strides,
                              self.pool_size, self.pool_strides)
        d_bias = d_y_conv.sum(axis=0)

        if self.l2reg_lambda > 0:
            d_w = d_w + self.filters * (self.l2reg_lambda / self.cached_input_shape[0])

        self.filters = (d_w * learning_rate).neg() + self.filters
        self.bias = ((d_bias * learning_rate).neg() + self.bias)

        return dx


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

    def initialize(self, input_shape, **kwargs):
        s = (input_shape[2] - self.pool_size[0]) // self.strides + 1
        return input_shape[:2] + [s, s]

    def forward(self,x):
        s = (x.shape[2] - self.pool_size[0]) // self.strides + 1
        self.initializer = type(x)
        pooled = self.initializer(np.zeros((x.shape[0], x.shape[1], s, s)))
        for j in range(s):
            for i in range(s):
                pooled[:, :, j, i] = x[:, :, j * self.strides:j * self.strides + self.pool_size[0],
                                       i * self.strides:i * self.strides + self.pool_size[1]].sum(axis=(2, 3))

        pooled = pooled / self.pool_area
        return pooled

    def backward(self, d_y, learning_rate):
        d_y_expanded = d_y.repeat(self.pool_size[0], axis=2)
        d_y_expanded = d_y_expanded.repeat(self.pool_size[1], axis=3)
        d_x = d_y_expanded / self.pool_area
        return d_x


class Reveal(Layer):

    def __init__(self):
        pass

    def initialize(self, input_shape, **kwargs):
        return input_shape

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
        loss = losses.sum(axis=0, keepdims=True).div(batch_size)
        return loss

    def derive(self, y_pred, y_correct):
        return y_correct


class SoftmaxCrossEntropy(Loss):
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

    def __init__(self, layers=[]):
        self.layers = layers

    def initialize(self, input_shape, initializer):
        for layer in self.layers:
            input_shape = layer.initialize(input_shape=input_shape, initializer= initializer, model=self)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_y, learning_rate):
        for layer in reversed(self.layers):
            d_y = layer.backward(d_y, learning_rate)

    @staticmethod
    def print_progress(batch_index, n_batches, batch_size, epoch_start, train_loss=None, train_acc=None,
                       val_loss=None, val_acc=None):
        sys.stdout.write('\r')
        sys.stdout.flush()
        progress = (batch_index / n_batches)

        eta = timedelta(seconds=round((1.-progress) * (time.time() - epoch_start) / progress, 0)) if progress > 0 else " "
        n_eq = int(progress * 30)
        n_dot = 30 - n_eq
        progress_bar = "=" * n_eq + ">" + n_dot * "."

        if val_loss is None:
            message = "{}/{} [{}] - ETA: {} - train_loss: {:.5f} - train_acc {:.5f}"
            sys.stdout.write(message.format((batch_index+1) * batch_size, n_batches * batch_size, progress_bar,
                                            eta, train_loss, train_acc))
        else:
            message = "{}/{} [{}] - ETA: {} - train_loss: {:.5f} - train_acc {:.5f} - val_loss {:.5f} - val_acc {:.5f}"
            sys.stdout.write(message.format((batch_index+1) * batch_size, n_batches * batch_size, progress_bar,
                                            eta, train_loss, train_acc, val_loss, val_acc))
        sys.stdout.flush()

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, loss=None, batch_size=32, epochs=1000,
            learning_rate=.01, verbose=0, eval_n_batches=None, results_file=None):

        if not isinstance(x_train, DataLoader): x_train = DataLoader(x_train)
        if not isinstance(y_train, DataLoader): y_train = DataLoader(y_train)

        if x_valid is not None:
            if not isinstance(x_valid, DataLoader): x_valid = DataLoader(x_valid)
            if not isinstance(y_train, DataLoader): y_valid = DataLoader(y_valid)

        n_batches = math.ceil(len(x_train.data) / batch_size)
        if eval_n_batches is None:
            eval_n_batches = n_batches

        for epoch in range(epochs):
            epoch_start = time.time()
            if verbose >= 1:
                print(datetime.now(), "Epoch {}/{}".format(epoch + 1, epochs))

            # Create batches on shuffled data
            shuffle = np.random.permutation(x_train.data.shape[0])
            batches = zip(x_train.batches(batch_size, shuffle_indices=shuffle),
                          y_train.batches(batch_size, shuffle_indices=shuffle))

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
                    if batch_index != 0 and (batch_index + 1) % eval_n_batches == 0:
                        # validation print
                        y_pred_val = self.predict(x_valid)
                        val_loss = np.sum(loss.evaluate(y_pred_val, y_valid.all_data()).unwrap())
                        val_acc = np.mean(y_valid.all_data().unwrap().argmax(axis=1) == y_pred_val.unwrap().argmax(axis=1))
                        self.print_progress(batch_index, n_batches, batch_size, epoch_start, train_acc=acc, train_loss=train_loss,
                                            val_loss=val_loss, val_acc=val_acc)
                    else:
                        # normal print
                        self.print_progress(batch_index, n_batches, batch_size, epoch_start, train_acc=acc, train_loss=train_loss)

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
