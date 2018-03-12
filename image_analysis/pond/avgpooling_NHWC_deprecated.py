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
