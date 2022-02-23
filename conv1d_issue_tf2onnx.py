from jinja2 import pass_context
import tensorflow as tf
import tf2onnx
import numpy as np

import keras2onnx
import onnx
from google.protobuf.json_format import MessageToDict

"""
python version: 3.8.10
tensorflow version: 2.2.0
tf2onnx version: 1.9.3
numpy version: 1.19.5
keras2onnx version: 1.7.0
onnx version 1.10.2
"""

def get_conv1d_flatten_dense(with_fancy_support = True, dump=True):
    model_input = tf.keras.layers.Input(shape=(10,6),batch_size=1)
    l1 = tf.keras.layers.Conv1D(5,3, activation="relu")
    l2 = tf.keras.layers.Flatten()
    model_output = tf.keras.layers.Dense(2, use_bias=False)
    model = tf.keras.models.Sequential([model_input,l1,l2,model_output])
    model.compile(loss='mean_absolute_error', optimizer='adam',
                  metrics=['mean_squared_error'])
    return model

if __name__ == "__main__":
    model = get_conv1d_flatten_dense()
    assert len(model.layers) == 3

    model_tfonnx, external_tensor_storage = tf2onnx.convert.from_keras(model,
                    input_signature=(tf.TensorSpec(model.input_shape,dtype=tf.dtypes.float32, name=None),), opset=None, custom_ops=None,
                    custom_op_handlers=None, custom_rewriter=None,
                    extra_opset=None, shape_override=None,
                    target=None, large_model=False, output_path=None) #, inputs_as_nchw="args_0")

    model_dict = MessageToDict(model_tfonnx)
    assert len(model_dict["graph"]["node"]) == 8   

    model_keras2onnx = keras2onnx.convert_keras(model, "my_conv1d_flatten_dense")

    model_dict_keras = MessageToDict(model_keras2onnx)
    assert len(model_dict_keras["graph"]["node"]) == 7

    for node in model_dict["graph"]["node"]: 
        print(node["opType"])
    for node in model_dict_keras["graph"]["node"]: 
        print(node["opType"])

    model.save(f"/tmp/conv1d_flatten_dense.h5")
    open("/tmp/conv1d_flatten_dense_tf2onnx.onnx", "wb").write(model_tfonnx.SerializeToString())
    open("/tmp/conv1d_flatten_dense_keras.onnx", "wb").write(model_keras2onnx.SerializeToString())

    pass 
