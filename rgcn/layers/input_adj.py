from keras.layers import Layer
from keras.engine.topology import Node
import keras.backend as K


def InputAdj(name=None, dtype=K.floatx(), sparse=False,
          tensor=None):
    shape = (None, None)
    input_layer = InputLayerAdj(batch_input_shape=shape,
                                name=name, sparse=sparse, input_dtype=dtype)
    outputs = input_layer.inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


class InputLayerAdj(Layer):
    def __init__(self, input_shape=None, batch_input_shape=None,
                 input_dtype=None, sparse=False, name=None):
        self.input_spec = None
        self.supports_masking = False
        self.uses_learning_phase = False
        self.trainable = False
        self.built = True

        self.inbound_nodes = []
        self.outbound_nodes = []

        self.trainable_weights = []
        self.non_trainable_weights = []
        self.constraints = {}

        self.sparse = sparse

        if not name:
            prefix = 'input'
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        if not batch_input_shape:
            assert input_shape, 'An Input layer should be passed either a `batch_input_shape` or an `input_shape`.'
            batch_input_shape = (None,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)
        if not input_dtype:
            input_dtype = K.floatx()

        self.batch_input_shape = batch_input_shape
        self.input_dtype = input_dtype

        input_tensor = K.placeholder(shape=batch_input_shape,
                                     dtype=input_dtype,
                                     sparse=self.sparse,
                                     name=self.name)

        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)
        shape = input_tensor._keras_shape
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[shape],
             output_shapes=[shape])

    def get_config(self):
        config = {'batch_input_shape': self.batch_input_shape,
                  'input_dtype': self.input_dtype,
                  'sparse': self.sparse,
                  'name': self.name}
        return config
