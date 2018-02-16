import numpy as np
from datetime import datetime
from functools import reduce


class Layer:
    pass


class Dense(Layer):
    
    def __init__(self, num_nodes, num_features, initial_scale=.01):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.initial_scale = initial_scale
        self.weights = None
        self.bias = None
        
    def initialize(self):
        self.weights = np.random.randn(self.num_features, self.num_nodes) * self.initial_scale
        self.bias    = np.zeros((1, self.num_nodes))
        
    def forward(self, x):
        y = x.dot(self.weights) + self.bias
        # cache result for backward pass
        self.cache = x
        return y

    def backward(self, d_y, learning_rate):
        x = self.cache
        # compute gradients for internal parameters and update
        d_weights = x.transpose().dot(d_y)
        d_bias = d_y.sum(axis=0)
        self.weights = (d_weights * learning_rate).neg() + self.weights
        self.bias    =    (d_bias * learning_rate).neg() + self.bias
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
        w0 =  0.5
        w1 =  0.2159198015
        w3 = -0.0082176259
        w5 =  0.0001825597
        w7 = -0.0000018848
        w9 =  0.0000000072
        
        x2 = x  * x
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
        likelihoods = x.exp()
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
        y = x.reshape([x.shape[0], -1])
        return y
    
    def backward(self, d_y, learning_rate):
        return d_y.reshape(self.shape)
                   


class Conv2D():
        
    def __init__(self, fshape, strides=1, filter_init=lambda shp: np.random.normal(scale=0.1, size=shp)):
        """ 2 Dimensional convolutional layer
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
                fmap[:, j, i, :] = (x[:, j * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis] * self.filters).sum(axis=(1, 2, 3))
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

    def forward(self,x):
        s = (x.shape[1] - self.pool_size[0]) // self.strides + 1
        self.initializer = type(x)
        pooled = self.initializer(np.zeros((x.shape[0], s, s, x.shape[3])))
        for j in range(s):
            for i in range(s):
                pooled[:, j, i, :] = x[:, j * self.strides:j * self.strides + self.pool_size[0], i * self.strides:i * self.strides + self.pool_size[1], :].sum(axis=(1, 2))
        
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
        losses = (probs_pred * probs_correct).log().neg()
        loss = losses.sum(axis=0).div(batch_size)
        return loss
        
    def derive(self, y_pred, y_correct):
        return y_correct


class DataLoader:
    
    def __init__(self, data, wrapper=lambda x: x):
        self.data = data
        self.wrapper = wrapper
        
    def batches(self, batch_size=None):
        if batch_size is None: batch_size = data.shape[0]
        return ( 
            self.wrapper(self.data[i:i+batch_size]) 
            for i in range(0, self.data.shape[0], batch_size) 
        )

    
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
        for layer in reversed(self.layers):
            d_y = layer.backward(d_y, learning_rate)
            
    def fit(self, x_train, y_train, loss, batch_size=32, epochs=1000, learning_rate=.01, verbose=0):
        if not isinstance(x_train, DataLoader): x_train = DataLoader(x_train)
        if not isinstance(y_train, DataLoader): y_train = DataLoader(y_train)
        for epoch in range(epochs):
            if verbose >= 1: print(datetime.now(), "Epoch %s" % epoch)
            batches = zip(x_train.batches(batch_size), y_train.batches(batch_size))
            for batch_index, (x_batch, y_batch) in enumerate(batches):
                if verbose >= 2: print(datetime.now(), "Batch %s" % batch_index)
                y_pred = self.forward(x_batch)
                d_y = loss.derive(y_pred, y_batch)
                self.backward(d_y, learning_rate)

    def predict(self, x, batch_size=32, verbose=0):
        if not isinstance(x, DataLoader): x = DataLoader(x)
        batches = []
        for batch_index, x_batch in enumerate(x.batches(batch_size)):
            if verbose >= 2: print(datetime.now(), "Batch %s" % batch_index)
            y_batch = self.forward(x_batch)
            batches.append(y_batch)
        return reduce(lambda x, y: x.concatenate(y), batches)
        
