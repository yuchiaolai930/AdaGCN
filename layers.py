import tensorflow as tf
import argparse

# Define parser for command-line flags
parser = argparse.ArgumentParser()
parser.add_argument('--hidden1', type=int, default=16, help='Number of units in hidden layer 1.')
parser.add_argument('--smoothing_steps', type=int, default=10, help='Number of smoothing steps for IGCN.')
parser.add_argument('--gnn', type=str, default='gcn', help='Type of GNN to use (gcn or igcn).')
FLAGS, unparsed = parser.parse_known_args()

# Global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    noise_shape = tf.cast(noise_shape, tf.int32)  # Ensure noise_shape is a tensor of int32
    if noise_shape.shape == ():  # If noise_shape is a scalar, convert it to a 1D tensor
        noise_shape = tf.expand_dims(noise_shape, 0)
    keep_prob = tf.cast(keep_prob, tf.float32)  # Ensure keep_prob is float32
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape, dtype=tf.float32)  # Ensure noise_shape is a vector
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return tf.sparse.SparseTensor(pre_out.indices, tf.cast(pre_out.values, tf.float32) * (1.0 / keep_prob), pre_out.dense_shape)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

class Dense(Layer):
    """Dense layer."""
    def __init__(self, 
                 placeholder_dropout, 
                 placeholder_num_features_nonzero,
                 weights,
                 bias,
                 dropout=True, 
                 sparse_inputs=False,
                 act=tf.nn.relu, 
                 flag_bias=False, 
                 **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholder_dropout
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.weights = weights
        self.bias = bias
        self.flag_bias = flag_bias

        # Helper variable for sparse dropout
        self.num_features_nonzero = placeholder_num_features_nonzero

    def _call(self, inputs):
        x = inputs

        # Dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # Transform
        output = dot(x, self.weights, sparse=self.sparse_inputs)

        # Bias
        if self.flag_bias:
            output += self.bias

        return self.act(output)

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 placeholder_dropout,
                 placeholder_support,
                 placeholder_num_features_nonzero,
                 weights,
                 bias,
                 dropout=True,
                 sparse_inputs=False, 
                 act=tf.nn.relu, 
                 flag_bias=False,
                 featureless=False, 
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholder_dropout
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholder_support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.flag_bias = flag_bias
        self.weights = weights
        self.bias = bias

        # Helper variable for sparse dropout
        self.num_features_nonzero = placeholder_num_features_nonzero

    def conv(self, adj, features):
        """IGCN renormalization filtering."""
        def tf_rnm(adj, features, k):
            new_feature = features
            for _ in range(k):
                new_feature = tf.sparse.sparse_dense_matmul(adj, new_feature)
            return new_feature

        result = tf_rnm(adj, features, FLAGS.smoothing_steps)
        return result

    def _call(self, inputs):
        x = inputs

        # Dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # Convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.weights, sparse=self.sparse_inputs)
            else:
                pre_sup = self.weights

            if FLAGS.gnn == 'gcn':
                support = dot(self.support[i], pre_sup, sparse=True)
            elif FLAGS.gnn == 'igcn':
                support = self.conv(self.support[i], pre_sup)
            supports.append(support)

        output = tf.add_n(supports)

        # Bias
        if self.flag_bias:
            output += self.bias

        return self.act(output)

class ImprovedGraphConvolution(GraphConvolution):
    """Improved Graph convolution layer with additional renormalization."""
    def __init__(self, **kwargs):
        super(ImprovedGraphConvolution, self).__init__(**kwargs)

    def conv(self, adj, features):
        """IGCN renormalization filtering with potential improvements."""
        def tf_rnm(adj, features, k):
            new_feature = features
            for _ in range(k):
                new_feature = tf.sparse.sparse_dense_matmul(adj, new_feature)
                new_feature = tf.nn.relu(new_feature)  # Additional non-linearity
            return new_feature

        result = tf_rnm(adj, features, FLAGS.smoothing_steps)
        return result
