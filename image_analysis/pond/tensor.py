import random
import numpy as np
from math import log
from im2col.im2col import im2col_indices, col2im_indices

try:
    from im2col.im2col_cython import im2col_cython, col2im_cython
    use_cython = True
except ImportError:
    print('\nRun the following from the image_analysis/im2col directory to use cython:')
    print('python setup.py build_ext --inplace\n')
    use_cython = False


class NativeTensor:

    def __init__(self, values):
        self.values = values

    def from_values(values):
        return NativeTensor(values)

    @property
    def size(self):
        return self.values.size

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index):
        return NativeTensor(self.values[index])

    def __setitem__(self, idx, other):
        assert isinstance(other, NativeTensor)
        self.values[idx] = other.values

    def concatenate(self, other):
        assert isinstance(other, NativeTensor), type(other)
        return NativeTensor.from_values(np.concatenate([self.values, other.values]))

    def reveal(self):
        return self

    def unwrap(self):
        return self.values

    def __repr__(self):
        return "NativeTensor(%s)" % self.values

    def wrap_if_needed(y):
        if isinstance(y, int) or isinstance(y, float): return NativeTensor.from_values(np.array([y]))
        if isinstance(y, np.ndarray): return NativeTensor.from_values(y)
        return y

    def add(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values + y.values)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).add(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).add(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    def __iadd__(self, y):
        if isinstance(y, NativeTensor): self.values = self.values + y.values
        elif isinstance(y, PublicEncodedTensor): self = PublicEncodedTensor.from_values(self.values).add(y)
        elif isinstance(y, PrivateEncodedTensor): self = PublicEncodedTensor.from_values(self.values).add(y)
        else: raise TypeError("does not support %s" % (type(y)))
        return self

    def add_at(self, indices, y):
        if isinstance(y, NativeTensor):
            np.add.at(self.values, indices, y.values)
        else:
            raise TypeError("%s does not support %s" % (type(self), type(y)))

    def sub(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values - y.values)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).sub(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).sub(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __sub__(x, y):
        return x.sub(y)

    def mul(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values * y.values)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).mul(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).mul(y)
        if isinstance(y, float): return NativeTensor(x.values * y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values.dot(y.values))
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).dot(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).dot(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def matmul(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(np.matmul(x.values, y.values))
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.values).matmul(y)
        if isinstance(y, PrivateEncodedTensor): return PublicEncodedTensor.from_values(x.values).matmul(y)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def div(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values / y.values)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __div__(x, y):
        return x.div(y)

    def __truediv__(x, y):
        return x.div(y)

    def __gt__(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values > y.values)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def pow(x, y):
        y = NativeTensor.wrap_if_needed(y)
        if isinstance(y, NativeTensor): return NativeTensor(x.values ** y.values)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __pow__(x, y):
        return x.pow(y)

    def transpose(x, *axes):
        if axes == ():
            return NativeTensor(x.values.transpose())
        else:
            return NativeTensor(x.values.transpose(axes))

    def neg(x):
        return NativeTensor(0 - x.values)

    def sum(x, axis=None, keepdims=False):
        return NativeTensor(x.values.sum(axis=axis, keepdims=keepdims))

    def clip(x, min, max):
        return NativeTensor.from_values(np.clip(x.values, min, max))

    def argmax(x, axis):
        return NativeTensor.from_values(x.values.argmax(axis=axis))

    def max(x, axis=None, keepdims=False):
        return NativeTensor.from_values(x.values.max(axis=axis, keepdims=keepdims))

    def min(x, axis=None, keepdims=False):
        return NativeTensor.from_values(x.values.min(axis=axis, keepdims=keepdims))

    def exp(x):
        return NativeTensor(np.exp(x.values))

    def log(x):
        # use this log to set log 0 -> -10^2
        return NativeTensor(np.ma.log(x.values).filled(-1e2))

    def inv(x):
        return NativeTensor(1. / x.values)

    def repeat(self, repeats, axis=None):
        self.values = np.repeat(self.values, repeats, axis=axis)
        return self

    def reshape(self, *shape):
        # ugly hack to unwrap shape if the shape is given as a tuple
        if isinstance(shape[0], tuple):
            shape = shape[0]
        return NativeTensor(self.values.reshape(shape))

    def pad(self, pad_width, mode='constant'):
        return NativeTensor(np.pad(self.values, pad_width=pad_width, mode=mode))

    def expand_dims(self, axis=0):
        self.values = np.expand_dims(self.values, axis=axis)
        return self

    def im2col(x, h_filter, w_filter, padding, strides):
        if use_cython:
            return NativeTensor(im2col_cython(x.values, h_filter, w_filter, padding, strides))
        else:
            return NativeTensor(im2col_indices(x.values, field_height=h_filter, field_width=w_filter, padding=padding,
                                               stride=strides))

    def col2im(x, imshape, field_height, field_width, padding, stride):
        if use_cython:
            return NativeTensor(col2im_cython(x.values, imshape[0], imshape[1], imshape[2], imshape[3],
                                            field_height, field_width, padding, stride))
        else:
            return NativeTensor(col2im_indices(x.values, imshape, field_height, field_width, padding, stride))

    def conv2d(x, filters, strides, padding):
        # shapes, assuming NCHW
        h_filter, w_filter, d_filters, n_filters = filters.shape
        n_x, d_x, h_x, w_x = x.shape
        h_out = int((h_x - h_filter + 2 * padding) / strides + 1)
        w_out = int((w_x - w_filter + 2 * padding) / strides + 1)

        # x to col
        X_col = x.im2col(h_filter, w_filter, padding, strides)
        W_col = filters.transpose(3, 2, 0, 1).reshape(n_filters, -1)
        out = W_col.dot(X_col)

        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out



DTYPE = 'object'
Q = 2657003489534545107915232808830590043

log2 = lambda x: log(x) / log(2)

# for arbitrary precision ints

# we need room for summing MAX_SUM values of MAX_DEGREE before during modulus reduction
MAX_DEGREE = 2
MAX_SUM = 2 ** 12
assert MAX_DEGREE * log2(Q) + log2(MAX_SUM) < 256

BASE = 2
PRECISION_INTEGRAL = 16
PRECISION_FRACTIONAL = 32
# TODO Gap as needed for local truncating

# we need room for double precision before truncating
assert PRECISION_INTEGRAL + 2 * PRECISION_FRACTIONAL < log(Q) / log(BASE)


def encode(rationals):
    # return (rationals * BASE ** PRECISION_FRACTIONAL).astype('int').astype(DTYPE) % Q
    try:
        return (rationals * BASE ** PRECISION_FRACTIONAL).astype('int').astype(DTYPE) % Q
    except OverflowError as e:
        print(rationals)
        raise e
        # print(e)
        # exit()


def decode(elements):
    try:
        map_negative_range = np.vectorize(lambda element: element if element <= Q / 2 else element - Q)
        return map_negative_range(elements) / BASE ** PRECISION_FRACTIONAL
    except OverflowError as e:
        print(elements)
        raise e


def wrap_if_needed(y):
    if isinstance(y, int) or isinstance(y, float): return PublicEncodedTensor.from_values(np.array([y]))
    if isinstance(y, np.ndarray):return PublicEncodedTensor.from_values(y)
    if isinstance(y, NativeTensor): return PublicEncodedTensor.from_values(y.values)
    return y


class PublicEncodedTensor:

    def __init__(self, values, elements=None):
        if not values is None:
            if not isinstance(values, np.ndarray):
                values = np.array([values])
            elements = encode(values)
        assert isinstance(elements, np.ndarray), "%s, %s, %s" % (values, elements, type(elements))
        self.elements = elements

    def from_values(values):
        return PublicEncodedTensor(values)

    def from_elements(elements):
        return PublicEncodedTensor(None, elements)

    def __repr__(self):
        return "PublicEncodedTensor(%s)" % decode(self.elements)

    def __getitem__(self, index):
        return PublicEncodedTensor.from_elements(self.elements[index])

    def __setitem__(self, idx, other):
        assert isinstance(other, PublicEncodedTensor)
        self.elements[idx] = other.elements

    def concatenate(self, other):
        assert isinstance(other, PublicEncodedTensor), type(other)
        return PublicEncodedTensor.from_elements(np.concatenate([self.elements, other.elements]))

    @property
    def shape(self):
        return self.elements.shape

    @property
    def size(self):
        return self.elements.size

    def unwrap(self):
        return decode(self.elements)

    def reveal(self):
        return NativeTensor.from_values(decode(self.elements))

    def truncate(self, amount=PRECISION_FRACTIONAL):
        positive_numbers = (self.elements <= Q // 2).astype(int)
        elements = self.elements
        elements = (Q + (2 * positive_numbers - 1) * elements) % Q  # x if x <= Q//2 else Q - x
        elements = np.floor_divide(elements, BASE ** amount)        # x // BASE**amount
        elements = (Q + (2 * positive_numbers - 1) * elements) % Q  # x if x <= Q//2 else Q - x
        return PublicEncodedTensor.from_elements(elements.astype(DTYPE))

    def add(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            return PublicEncodedTensor.from_elements((x.elements + y.elements) % Q)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.elements + y.shares0) % Q
            shares1 = y.shares1
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    # def add_at(x, indices, y):
    #
    #     y = wrap_if_needed(y)
    #     if isinstance(y, PublicEncodedTensor):
    #         np.add.at(x.elements, indices, y.elements)
    #     # if isinstance(y, PrivateEncodedTensor):
    #     #     shares0 = np.add.at(x.elements, indices, y.shares0) % Q
    #     #     shares1 = y.shares1
    #     #     return PrivateEncodedTensor.from_shares(shares0, shares1)
    #     else:
    #         raise TypeError("%s does not support %s" % (type(x), type(y)))

    def sub(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_elements((x.elements - y.elements) % Q)
        if isinstance(y, PrivateEncodedTensor): return x.add(y.neg())  # TODO there might be a more efficient way
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __sub__(x, y):
        return x.sub(y)

    def __gt__(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_values(x.elements > y.elements)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def mul(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicFieldTensor):
            return PublicFieldTensor.from_elements((x.elements * y.elements) % Q)
        if isinstance(y, PublicEncodedTensor):
            return PublicEncodedTensor.from_elements((x.elements * y.elements) % Q).truncate()
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.elements * y.shares0) % Q
            shares1 = (x.elements * y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor): return PublicEncodedTensor.from_elements(
            x.elements.dot(y.elements) % Q).truncate()
        if isinstance(y, PrivateEncodedTensor):
            shares0 = x.elements.dot(y.shares0) % Q
            shares1 = x.elements.dot(y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def div(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, NativeTensor): return x.mul(y.inv())
        if isinstance(y, PublicEncodedTensor): return x.mul(y.inv())
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def matmul(x, y):
        return PublicEncodedTensor(np.matmul(x.elements, y.elements))

    def __div__(x, y):
        return x.div(y)

    def __truediv__(x, y):
        return x.div(y)

    def transpose(x, *axes):
        if axes == ():
            return PublicEncodedTensor.from_elements(x.elements.transpose())
        else:
            return PublicEncodedTensor.from_elements(x.elements.transpose(axes))

    def sum(x, axis=None, keepdims=False):
        return PublicEncodedTensor.from_elements(x.elements.sum(axis=axis, keepdims=keepdims))

    def argmax(x, axis):
        return PublicEncodedTensor.from_values(decode(x.elements).argmax(axis=axis))

    def neg(x):
        return PublicEncodedTensor.from_values(decode(x.elements) * -1)

    def inv(x):
        return PublicEncodedTensor.from_values(1. / decode(x.elements))

    def repeat(self, repeats, axis=None):
        self.elements = np.repeat(self.elements, repeats, axis=axis)
        return self

    def reshape(self, *shape):
        # ugly hack to unwrap shape if the shape is given as a tuple
        if isinstance(shape[0], tuple):
            shape = shape[0]
        return PublicEncodedTensor.from_elements(self.elements.reshape(shape))

    def pad(self, pad_width, mode='constant'):
        return PublicEncodedTensor.from_elements(np.pad(self.elements, pad_width=pad_width, mode=mode))

    def im2col(x, h_filter, w_filter, padding, strides):
        if use_cython:
            return PublicEncodedTensor.from_elements(im2col_cython(x.elements.astype('float'), h_filter, w_filter, padding,
                                                                   strides).astype('int').astype(DTYPE))
        else:
            return PublicEncodedTensor.from_elements(im2col_indices(x.elements.astype('float'),
                                                                    field_height=h_filter, field_width=w_filter,
                                                                    padding=padding,stride=strides).astype('int').astype(DTYPE))

    def col2im(x, imshape, field_height, field_width, padding, stride):
        if use_cython:
            return PublicEncodedTensor.from_elements(col2im_cython(x.elements.astype('float'), imshape[0], imshape[1], imshape[2], imshape[3],
                                              field_height, field_width, padding, stride).astype('int').astype(DTYPE))
        else:
            return PublicEncodedTensor.from_elements(col2im_indices(x.elements.astype('float'), imshape, field_height,
                                                                    field_width, padding, stride).astype('int').astype(DTYPE))


    def conv2d(x, filters, strides, padding):
        # shapes, assuming NCHW
        h_filter, w_filter, d_filters, n_filters = filters.shape
        n_x, d_x, h_x, w_x = x.shape
        h_out = int((h_x - h_filter + 2 * padding) / strides + 1)
        w_out = int((w_x - w_filter + 2 * padding) / strides + 1)

        # x to col
        X_col = x.im2col(h_filter, w_filter, padding, strides)
        W_col = filters.transpose(3, 2, 0, 1).reshape(n_filters, -1)
        out = W_col.dot(X_col)

        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out


class PublicFieldTensor:

    def __init__(self, elements):
        self.elements = elements

    def from_elements(elements):
        return PublicFieldTensor(elements)

    def __repr__(self):
        return "PublicFieldTensor(%s)" % self.elements

    def __getitem__(self, index):
        return PublicFieldTensor.from_elements(self.elements[index])

    @property
    def size(self):
        return self.elements.size

    @property
    def shape(self):
        return self.elements.shape

    def add(x, y):
        if isinstance(y, PublicFieldTensor):
            return PublicFieldTensor.from_elements((x.elements + y.elements) % Q)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.elements + y.shares0) % Q
            shares1 = y.shares1
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    def mul(x, y):
        if isinstance(y, PublicFieldTensor):
            return PublicFieldTensor.from_elements((x.elements * y.elements) % Q)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.elements * y.shares0) % Q
            shares1 = (x.elements * y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y):
        if isinstance(y, PublicFieldTensor):
            return PublicFieldTensor.from_elements((x.elements.dot(y.elements)) % Q)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.elements.dot(y.shares0)) % Q
            shares1 = (x.elements.dot(y.shares1)) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))


    def transpose(x, *axes):
        if axes == ():
            return PublicFieldTensor.from_elements(x.elements.transpose())
        else:
            return PublicFieldTensor.from_elements(x.elements.transpose(axes))

    def reshape(self, *shape):
        # ugly hack to unwrap shape if the shape is given as a tuple
        if isinstance(shape[0], tuple):
            shape = shape[0]
        return PublicFieldTensor.from_elements(self.elements.reshape(shape))


def share(elements):
    shares0 = np.array([random.randrange(Q) for _ in range(elements.size)]).astype(DTYPE).reshape(elements.shape)
    shares1 = ((elements - shares0) % Q).astype(DTYPE)
    return shares0, shares1


def reconstruct(shares0, shares1):
    return (shares0 + shares1) % Q


class PrivateFieldTensor:

    def __init__(self, elements, shares0=None, shares1=None):
        if not elements is None:
            shares0, shares1 = share(elements)
        assert isinstance(shares0, np.ndarray), "%s, %s, %s" % (values, shares0, type(shares0))
        assert isinstance(shares1, np.ndarray), "%s, %s, %s" % (values, shares1, type(shares1))
        assert shares0.shape == shares1.shape
        self.shares0 = shares0
        self.shares1 = shares1

    def from_elements(elements):
        return PrivateFieldTensor(elements)

    def from_shares(shares0, shares1):
        return PrivateFieldTensor(None, shares0, shares1)

    def reveal(self):
        return PublicFieldTensor.from_elements(reconstruct(self.shares0, self.shares1))

    def __repr__(self):
        return "PrivateFieldTensor(%s)" % self.reveal().elements

    def __getitem__(self, index):
        return PrivateFieldTensor.from_shares(self.shares0[index], self.shares1[index])

    @property
    def size(self):
        return self.shares0.size

    @property
    def shape(self):
        return self.shares0.shape

    def add(x, y):
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0 + y.shares0) % Q
            shares1 = (x.shares1 + y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0 + y.elements) % Q
            shares1 = x.shares1
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    def mul(x, y):
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0 * y.elements) % Q
            shares1 = (x.shares1 * y.elements) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y):
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0.dot(y.elements)) % Q
            shares1 = (x.shares1.dot(y.elements)) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def transpose(x, *axes):
        if axes == ():
            return PrivateFieldTensor.from_shares(x.shares0.transpose(), x.shares1.transpose())
        else:
            return PrivateFieldTensor.from_shares(x.shares0.transpose(axes), x.shares1.transpose(axes))

    def reshape(self, *shape):
        # ugly hack to unwrap shape if the shape is given as a tuple
        if isinstance(shape[0], tuple):
            shape = shape[0]
        return PrivateFieldTensor.from_shares(self.shares0.reshape(shape), self.shares1.reshape(shape))

    def conv2d(x, y, strides, padding):
        if isinstance(y, PublicFieldTensor):

            # shapes, assuming NCHW
            h_filter, w_filter, d_filters, n_filters = y.shape
            n_x, d_x, h_x, w_x = x.shape
            h_out = int((h_x - h_filter + 2 * padding) / strides + 1)
            w_out = int((w_x - w_filter + 2 * padding) / strides + 1)

            # x to col
            X_col = x.im2col(h_filter, w_filter, padding, strides)
            W_col = y.transpose(3, 2, 0, 1).reshape(n_filters, -1)

            out = W_col.dot(X_col)
            out = out.reshape(n_filters, h_out, w_out, n_x)
            out = out.transpose(3, 0, 1, 2)
            return out

        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def im2col(x, h_filter, w_filter, padding, strides):
        if use_cython:
            shares0 = im2col_cython(x.shares0.astype('float'), h_filter, w_filter, padding,
                                    strides).astype('int').astype(DTYPE)
            shares1 = im2col_cython(x.shares1.astype('float'), h_filter, w_filter, padding,
                                    strides).astype('int').astype(DTYPE)
            return PrivateFieldTensor.from_shares(shares0, shares1)
        else:
            shares0 = im2col_indices(x.shares0.astype('float'), field_height=h_filter, field_width=w_filter,
                                     padding=padding,stride=strides).astype('int').astype(DTYPE)
            shares1 = im2col_indices(x.shares1.astype('float'), field_height=h_filter, field_width=w_filter,
                                     padding=padding,stride=strides).astype('int').astype(DTYPE)
            return PrivateFieldTensor.from_shares(shares0, shares1)


def generate_mul_triple(shape):
    a = np.array([random.randrange(Q) for _ in range(np.prod(shape))]).astype(DTYPE).reshape(shape)
    b = np.array([random.randrange(Q) for _ in range(np.prod(shape))]).astype(DTYPE).reshape(shape)
    ab = (a * b) % Q
    return PrivateFieldTensor.from_elements(a), \
           PrivateFieldTensor.from_elements(b), \
           PrivateFieldTensor.from_elements(ab)


def generate_dot_triple(m, n, o):
    a = np.array([random.randrange(Q) for _ in range(m * n)]).astype(DTYPE).reshape((m, n))
    b = np.array([random.randrange(Q) for _ in range(n * o)]).astype(DTYPE).reshape((n, o))
    ab = np.dot(a, b)
    return PrivateFieldTensor.from_elements(a), \
           PrivateFieldTensor.from_elements(b), \
           PrivateFieldTensor.from_elements(ab)

def generate_conv_triple(xshape, yshape, strides, padding):

    h_filter, w_filter, d_filters, n_filters = yshape
    n_x, d_x, h_x, w_x = xshape
    h_out = int((h_x - h_filter + 2 * padding) / strides + 1)
    w_out = int((w_x - w_filter + 2 * padding) / strides + 1)

    a = np.array([random.randrange(Q) for _ in range(np.prod(xshape))]).astype(DTYPE).reshape(xshape)
    b = np.array([random.randrange(Q) for _ in range(np.prod(yshape))]).astype(DTYPE).reshape(yshape)

    if use_cython:
        a_col = im2col_cython(a, h_filter, w_filter, padding, strides)
    else:
        a_col = im2col_indices(a, field_height=h_filter, field_width=w_filter, padding=padding, stride=strides)
    b_col = b.transpose(3, 2, 0, 1).reshape(n_filters, -1)

    a_conv_b = b_col.dot(a_col)
    a_conv_b = a_conv_b.reshape(n_filters, h_out, w_out, n_x)
    a_conv_b = a_conv_b.transpose(3, 0, 1, 2)
    return a, b, a_conv_b


def generate_matmul_triple(shape1, shape2):
    a = np.array([random.randrange(Q) for _ in range(m * n)]).astype(DTYPE).reshape(shape1)
    b = np.array([random.randrange(Q) for _ in range(n * o)]).astype(DTYPE).reshape(shape2)
    ab = np.matmul(a, b)
    return PrivateFieldTensor.from_elements(a), \
           PrivateFieldTensor.from_elements(b), \
           PrivateFieldTensor.from_elements(ab)


# def generate_mul_triple(shape):
#     a = np.zeros(shape).astype(int).astype(DTYPE)
#     b = np.zeros(shape).astype(int).astype(DTYPE)
#     ab = (a * b) % Q
#     return PrivateFieldTensor.from_elements(a), \
#            PrivateFieldTensor.from_elements(b), \
#            PrivateFieldTensor.from_elements(ab)
# 
# def generate_dot_triple(m, n, o):
#     a = np.zeros((m,n)).astype(int).astype(DTYPE)
#     b = np.zeros((n,o)).astype(int).astype(DTYPE)
#     ab = np.dot(a, b)
#     return PrivateFieldTensor.from_elements(a), \
#            PrivateFieldTensor.from_elements(b), \
#            PrivateFieldTensor.from_elements(ab)


class PrivateEncodedTensor:

    def __init__(self, values, shares0=None, shares1=None):
        if not values is None:
            if not isinstance(values, np.ndarray):
                values = np.array([values])
            shares0, shares1 = share(encode(values))
        assert isinstance(shares0, np.ndarray), "%s, %s, %s" % (values, shares0, type(shares0))
        assert isinstance(shares1, np.ndarray), "%s, %s, %s" % (values, shares1, type(shares1))
        assert shares0.dtype == shares1.dtype
        assert shares0.shape == shares1.shape
        self.shares0 = shares0
        self.shares1 = shares1

    def from_values(values):
        return PrivateEncodedTensor(values)

    def from_elements(elements):
        shares0, shares1 = share(elements)
        return PrivateEncodedTensor(None, shares0, shares1)

    def from_shares(shares0, shares1):
        return PrivateEncodedTensor(None, shares0, shares1)

    def __repr__(self):
        elements = (self.shares0 + self.shares1) % Q
        return "PrivateEncodedTensor(%s)" % decode(elements)

    def __getitem__(self, index):
        return PrivateEncodedTensor.from_shares(self.shares0[index], self.shares1[index])

    def __setitem__(self, idx, other):
        assert isinstance(other, PrivateEncodedTensor)
        self.shares0[idx] = other.shares0
        self.shares1[idx] = other.shares1

    def concatenate(self, other):
        assert isinstance(other, PrivateEncodedTensor), type(other)
        shares0 = np.concatenate([self.shares0, other.shares0])
        shares1 = np.concatenate([self.shares1, other.shares1])
        return PrivateEncodedTensor.from_shares(shares0, shares1)

    @property
    def shape(self):
        return self.shares0.shape

    @property
    def size(self):
        return self.shares0.size

    def unwrap(self):
        return decode((self.shares0 + self.shares1) % Q)

    def reveal(self):
        return NativeTensor.from_values(decode((self.shares0 + self.shares1) % Q))

    def truncate(self, amount=PRECISION_FRACTIONAL):
        shares0 = np.floor_divide(self.shares0, BASE ** amount) % Q
        shares1 = (Q - (np.floor_divide(Q - self.shares1, BASE ** amount))) % Q
        return PrivateEncodedTensor.from_shares(shares0, shares1)

    def add(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = (x.shares0 + y.elements) % Q
            shares1 = x.shares1
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.shares0 + y.shares0) % Q
            shares1 = (x.shares1 + y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __add__(x, y):
        return x.add(y)

    def add_at(x, indices, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            np.add.at(x.shares0, indices, y.elements)
            x.shares0 = x.shares0 % Q
        elif isinstance(y, PrivateEncodedTensor):
            np.add.at(x.shares0, indices, y.shares0)
            np.add.at(x.shares1, indices, y.shares1)
            x.shares0 = x.shares0 % Q
            x.shares1 = x.shares1 % Q
        else:
            raise TypeError("%s does not support %s" % (type(x), type(y)))

    def sub(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = (x.shares0 - y.elements) % Q
            shares1 = x.shares1
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateEncodedTensor):
            shares0 = (x.shares0 - y.shares0) % Q
            shares1 = (x.shares1 - y.shares1) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0 - y.shares0) % Q
            shares1 = (x.shares1 - y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __sub__(x, y):
        return x.sub(y)

    def mul(x, y, precomputed=None):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = (x.shares0 * y.elements) % Q
            shares1 = (x.shares1 * y.elements) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        if isinstance(y, PrivateEncodedTensor):
            x_broadcasted0, y_broadcasted0 = np.broadcast_arrays(x.shares0, y.shares0)
            x_broadcasted1, y_broadcasted1 = np.broadcast_arrays(x.shares1, y.shares1)
            x_broadcasted = PrivateEncodedTensor.from_shares(x_broadcasted0, x_broadcasted1)
            y_broadcasted = PrivateEncodedTensor.from_shares(y_broadcasted0, y_broadcasted1)

            if precomputed is None: precomputed = generate_mul_triple(x_broadcasted.shape)
            a, b, ab = precomputed
            assert x_broadcasted.shape == y_broadcasted.shape
            assert x_broadcasted.shape == a.shape
            assert y_broadcasted.shape == b.shape
            alpha = (x_broadcasted - a).reveal()
            beta = (y_broadcasted - b).reveal()
            z = alpha.mul(beta) + \
                alpha.mul(b) + \
                a.mul(beta) + \
                ab
            return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()
        if isinstance(y, PrivateFieldTensor):
            shares0 = (x.shares0 * y.shares0) % Q
            shares1 = (x.shares1 * y.shares1) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        if isinstance(y, PublicFieldTensor):
            shares0 = (x.shares0 * y.elements) % Q
            shares1 = (x.shares1 * y.elements) % Q
            return PrivateFieldTensor.from_shares(shares0, shares1)
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y, precomputed=None):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            assert x.shape[1] == y.shape[0]
            shares0 = x.shares0.dot(y.elements) % Q
            shares1 = x.shares1.dot(y.elements) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        if isinstance(y, PrivateEncodedTensor):
            m = x.shape[0]
            n = x.shape[1]
            o = y.shape[1]
            assert n == y.shape[0]
            if precomputed is None: precomputed = generate_dot_triple(m, n, o)
            a, b, ab = precomputed
            alpha = (x - a).reveal() # (PrivateEncodedTensor - PrivateFieldTensor).reveal() = PublicFieldTensor
            beta = (y - b).reveal()  # (PrivateEncodedTensor - PrivateFieldTensor).reveal() = PublicFieldTensor
            z = alpha.dot(beta) + alpha.dot(b) + a.dot(beta) + ab
            # PublicFieldTensor.dot(PublicFieldTensor) = PublicFieldTensor
            # PublicFieldTensor.dot(PrivateFieldTensor) = PrivateFieldTensor
            # PrivateFieldTensor.dot(PublicFieldTensor) = PrivateFieldTensor
            # PrivateFieldTensor = PrivateFieldTensor
            # PublicFieldTensor + PrivateFieldTensor + PrivateFieldTensor + PrivateFieldTensor = PrivateFieldTensor
            return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def div(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, NativeTensor): return x.mul(y.inv())
        if isinstance(y, PublicEncodedTensor): return x.mul(y.inv())
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def matmul(x, y):
        y = wrap_if_needed(y)
        if isinstance(y, PublicEncodedTensor):
            shares0 = np.matmul(x.shares0, y.elements) % Q
            shares1 = np.matmul(x.shares1, y.elements) % Q
            return PrivateEncodedTensor.from_shares(shares0, shares1).truncate()
        if isinstance(y, PrivateEncodedTensor):
            if precomputed is None: precomputed = generate_matmul_triple(x.shape, y.shape)
            a, b, ab = precomputed
            alpha = (x - a).reveal()
            beta = (y - b).reveal()
            z = np.matmul(alpha, beta) + \
                np.matmul(alpha, b) + \
                np.matmul(a, beta) + \
                ab
            return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()
        raise TypeError("%s does not support %s" % (type(x), type(y)))

    def __truediv__(x, y):
        return x.div(y)

    def neg(self):
        minus_one = PublicFieldTensor.from_elements(np.array([Q - 1]))
        z = self.mul(minus_one)
        return PrivateEncodedTensor.from_shares(z.shares0, z.shares1)

    def transpose(self, *axes):
        if axes == ():
            return PrivateEncodedTensor.from_shares(self.shares0.transpose(), self.shares1.transpose())
        else:
            return PrivateEncodedTensor.from_shares(self.shares0.transpose(axes), self.shares1.transpose(axes))

    def sum(self, axis, keepdims=False):
        shares0 = self.shares0.sum(axis=axis, keepdims=keepdims) % Q
        shares1 = self.shares1.sum(axis=axis, keepdims=keepdims) % Q
        return PrivateEncodedTensor.from_shares(shares0, shares1)

    def repeat(self, repeats, axis=None):
        self.shares0 = np.repeat(self.shares0, repeats, axis=axis)
        self.shares1 = np.repeat(self.shares1, repeats, axis=axis)
        return self

    def reshape(self, *shape):
        # ugly hack to unwrap shape if the shape is given as a tuple
        if isinstance(shape[0], tuple):
            shape = shape[0]
        return PrivateEncodedTensor.from_shares(self.shares0.reshape(shape), self.shares1.reshape(shape))

    def pad(self, pad_width, mode='constant'):
        return PrivateEncodedTensor.from_shares(np.pad(self.shares0, pad_width=pad_width, mode=mode),
                                                np.pad(self.shares1, pad_width=pad_width, mode=mode))


    def im2col(x, h_filter, w_filter, padding, strides):
        if use_cython:
            shares0 = im2col_cython(x.shares0.astype('float'), h_filter, w_filter, padding,
                                    strides).astype('int').astype(DTYPE)
            shares1 = im2col_cython(x.shares1.astype('float'), h_filter, w_filter, padding,
                                    strides).astype('int').astype(DTYPE)
            return PrivateEncodedTensor.from_shares(shares0, shares1)
        else:
            shares0 = im2col_indices(x.shares0.astype('float'), field_height=h_filter, field_width=w_filter,
                                     padding=padding,stride=strides).astype('int').astype(DTYPE)
            shares1 = im2col_indices(x.shares1.astype('float'), field_height=h_filter, field_width=w_filter,
                                     padding=padding,stride=strides).astype('int').astype(DTYPE)
            return PrivateEncodedTensor.from_shares(shares0, shares1)


    def col2im(x, imshape, field_height, field_width, padding, stride):
        if use_cython:
            shares0 =col2im_cython(x.shares0.astype('float'), imshape[0], imshape[1], imshape[2], imshape[3],
                                   field_height, field_width, padding, stride).astype('int').astype(DTYPE)
            shares1 =col2im_cython(x.shares1.astype('float'), imshape[0], imshape[1], imshape[2], imshape[3],
                                   field_height, field_width, padding, stride).astype('int').astype(DTYPE)

            return PrivateEncodedTensor.from_shares(shares0, shares1)
        else:
            shares0 = col2im_indices(x.shares0.astype('float'), imshape, field_height, field_width, padding,
                                     stride).astype('int').astype(DTYPE)
            shares1 = col2im_indices(x.shares1.astype('float'), imshape, field_height, field_width, padding,
                                     stride).astype('int').astype(DTYPE)
            return PrivateEncodedTensor.from_shares(shares0, shares1)


    def conv2d(x, filters, strides, padding, precomputed=None, use_specialized_triple=True):
        h_filter, w_filter, d_filters, n_filters = filters.shape
        n_x, d_x, h_x, w_x = x.shape
        h_out = int((h_x - h_filter + 2 * padding) / strides + 1)
        w_out = int((w_x - w_filter + 2 * padding) / strides + 1)

        if isinstance(filters, PublicEncodedTensor) or (isinstance(filters, PrivateEncodedTensor) and not use_specialized_triple):
            X_col = x.im2col(h_filter, w_filter, padding, strides)
            W_col = filters.transpose(3, 2, 0, 1).reshape(n_filters, -1)
            out = W_col.dot(X_col).reshape(n_filters, h_out, w_out, n_x).transpose(3, 0, 1, 2)
            return out

        if isinstance(filters, PrivateEncodedTensor):
            if use_specialized_triple:

                if precomputed is None:
                    precomputed = generate_conv_triple(x.shape, filters.shape, strides, padding)
                a, b, a_conv_b = precomputed          # PrivateFieldTensors
                alpha = (x - a).reveal()        # (PrivateEncodedTensor - PrivateFieldTensor).reveal() = PublicFieldTensor
                beta = (filters - b).reveal()   # (PrivateEncodedTensor - PrivateFieldTensor).reveal() = PublicFieldTensor

                alpha_col = alpha.im2col(h_filter, w_filter, padding, strides) # PublicFieldTensor
                alpha_conv_beta = alpha_col.dot(beta).transpose(3, 2, 0, 1).reshape(n_filters, -1)
                alpha_conv_b = alpha_col.dot(b).transpose(3, 2, 0, 1).reshape(n_filters, -1)

                z = alpha_conv_beta + alpha_conv_b + a.conv2d(beta) + a_conv_b

                return PrivateEncodedTensor.from_shares(z.shares0, z.shares1).truncate()
                # PublicFieldTensor.dot(PublicFieldTensor) = PublicFieldTensor
                # PublicFieldTensor.dot(PrivateFieldTensor) = PrivateFieldTensor
                # PrivateFieldTensor.conv2d(PublicFieldTensor) = PrivateFieldTensor
                # PrivateFieldTensor = PrivateFieldTensor
                # PublicFieldTensor + PrivateFieldTensor + PrivateFieldTensor + PrivateFieldTensor = PrivateFieldTensor

        raise TypeError("%s does not support %s" % (type(x), type(filters)))


ANALYTIC_STORE = []
NEXT_ID = 0


class AnalyticTensor:

    def __init__(self, values, shape=None, ident=None):
        if not values is None:
            if not isinstance(values, np.ndarray):
                values = np.array([values])
            shape = values.shape
        if ident is None:
            global NEXT_ID
            ident = "tensor_%d" % NEXT_ID
            NEXT_ID += 1
        self.shape = shape
        self.ident = ident

    def from_shape(shape, ident=None):
        return AnalyticTensor(None, shape, ident)

    def __repr__(self):
        return "AnalyticTensor(%s, %s)" % (self.shape, self.ident)

    def __getitem__(self, index):
        start, stop, _ = index.indices(self.shape[0])
        shape = list(self.shape)
        shape[0] = stop - start
        ident = "%s_%d,%d" % (self.ident, start, stop)
        return AnalyticTensor.from_shape(tuple(shape), ident)

    def reset():
        global ANALYTIC_STORE
        ANALYTIC_STORE = []

    def store():
        global ANALYTIC_STORE
        return ANALYTIC_STORE

    @property
    def size(self):
        return np.prod(self.shape)

    def reveal(self):
        return self

    def wrap_if_needed(y):
        if isinstance(y, int) or isinstance(y, float): return AnalyticTensor.from_shape((1,))
        if isinstance(y, np.ndarray): return AnalyticTensor.from_shape(y.shape)
        return y

    def add(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('add', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def __add__(x, y):
        return x.add(y)

    def sub(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('sub', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def __sub__(x, y):
        return x.sub(y)

    def mul(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('mul', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def __mul__(x, y):
        return x.mul(y)

    def dot(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('dot', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def div(x, y):
        y = AnalyticTensor.wrap_if_needed(y)
        ANALYTIC_STORE.append(('div', x, y))
        return AnalyticTensor.from_shape(x.shape)

    def neg(self):
        ANALYTIC_STORE.append(('neg', self))
        return AnalyticTensor.from_shape(self.shape)

    def transpose(self):
        ANALYTIC_STORE.append(('transpose', self))
        return self

    def sum(self, axis):
        ANALYTIC_STORE.append(('sum', self))
        return AnalyticTensor.from_shape(self.shape)
