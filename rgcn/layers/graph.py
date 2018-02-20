from __future__ import print_function

from keras import activations, initializations
from keras import regularizers
from keras.engine import Layer
from keras.layers import Dropout

import keras.backend as K


class GraphConvolution(Layer):
    def __init__(self, output_dim, support=1, featureless=False,
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, num_bases=-1,
                 b_regularizer=None, bias=False, dropout=0., **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights
        self.featureless = featureless  # use/ignore input features
        self.dropout = dropout

        assert support >= 1

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights
        self.num_bases = num_bases

        # these will be defined during build()
        self.input_dim = None
        self.W = None
        self.W_comp = None
        self.b = None
        self.num_nodes = None

        super(GraphConvolution, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.output_dim)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        if self.featureless:
            self.num_nodes = features_shape[1]  # NOTE: Assumes featureless input (i.e. square identity mx)
        assert len(features_shape) == 2
        self.input_dim = features_shape[1]

        if self.num_bases > 0:
            self.W = K.concatenate([self.add_weight((self.input_dim, self.output_dim),
                                                    initializer=self.init,
                                                    name='{}_W'.format(self.name),
                                                    regularizer=self.W_regularizer) for _ in range(self.num_bases)],
                                   axis=0)

            self.W_comp = self.add_weight((self.support, self.num_bases),
                                          initializer=self.init,
                                          name='{}_W_comp'.format(self.name),
                                          regularizer=self.W_regularizer)
        else:
            self.W = K.concatenate([self.add_weight((self.input_dim, self.output_dim),
                                                    initializer=self.init,
                                                    name='{}_W'.format(self.name),
                                                    regularizer=self.W_regularizer) for _ in range(self.support)],
                                   axis=0)

        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, inputs, mask=None):
        features = inputs[0]
        A = inputs[1:]  # list of basis functions

        # convolve
        supports = list()
        for i in range(self.support):
            if not self.featureless:
                supports.append(K.dot(A[i], features))
            else:
                supports.append(A[i])
        supports = K.concatenate(supports, axis=1)

        if self.num_bases > 0:
            self.W = K.reshape(self.W,
                               (self.num_bases, self.input_dim, self.output_dim))
            self.W = K.permute_dimensions(self.W, (1, 0, 2))
            V = K.dot(self.W_comp, self.W)
            V = K.reshape(V, (self.support*self.input_dim, self.output_dim))
            output = K.dot(supports, V)
        else:
            output = K.dot(supports, self.W)

        # if featureless add dropout to output, by elementwise multiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        if self.featureless:
            tmp = K.ones(self.num_nodes)
            tmp_do = Dropout(self.dropout)(tmp)
            output = (output.T * tmp_do).T

        if self.bias:
            output += self.b
        return self.activation(output)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'num_bases': self.num_bases,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))