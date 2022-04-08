# med2fe, PS-EC/ESD
#
# Converter should reveal the lack of support and inpotablility between pytorch and tensorfow. And impact on compatibility using onnx
# it should clarify, that just an exchange format does not change the required effort, if different frameworks are supported. Conversions 
# are a very bad choice.
# 
# Reasoning is based on the Convolutional:
#   the storage order order defines 
# 
# 
# data_format is defined as storage order in onnx - but not supported vor Conv operator
#                                                   https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Conv-11
# Tensorflow focuses only on data_format = channel_last namely (NWC, NWHC), onnx supports only NCW and NCHW is the implementation in pytorch.
#  
 
import tensorflow as tf
import numpy as np
import os, sys
import onnx
from onnx import helper as h
from onnx import TensorProto as tp
from onnx import checker
from onnx.checker import ValidationError
from onnx import save
import json
from google.protobuf.json_format import MessageToDict
import numpy as np
import onnxruntime as rt

root_dir = f"{os.path.dirname(os.path.abspath(__file__))}/.."
temp_dir = f"{os.path.dirname(os.path.abspath(__file__))}/outputs"
image_folder = f"{root_dir}/tests/images/"

def gen_conv1d_flatten_dense(strides = 1, dilation_rate = 1, kernel_size = 3, 
                input_shape = (200,10,20), num_filter = 1, padding="valid",
                data_format = "channels_last"):
    np.random.seed(0)    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(num_filter, kernel_size, input_shape=input_shape[1:],strides=strides, 
                                        dilation_rate = dilation_rate, 
                                        padding=padding, batch_size=1,data_format=data_format)) #,data_format='channels_last'))
    model.add(tf.keras.layers.Conv1D(num_filter, kernel_size,strides=strides, 
                                        dilation_rate = dilation_rate, 
                                        padding=padding,data_format=data_format))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Dense(2))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])

    nweights = []
    for i,w in enumerate(model.get_weights()):
        c = 1
        for n in w.shape: 
            c *= n
        nweights.append(np.arange(-1, c-1).reshape(w.shape))
    model.set_weights(nweights)
    model.save(f"{temp_dir}/ps-et_anaomaly_detection.h5")
    return model

def gen_dense(input_shape = (1,10)):
    np.random.seed(0)    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(15, input_shape=input_shape[1:],batch_size=1))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dense(2))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])

    nweights = []
    for i,w in enumerate(model.get_weights()):
        c = 1
        for n in w.shape: 
            c *= n
        nweights.append(np.arange(-1, c-1).reshape(w.shape))
    model.set_weights(nweights)
    return model

def make_conv1d_node(layer):
    config = layer.get_config()
    a_kernel_shape=h.make_attribute("kernel_shape", config['kernel_size'])
    a_dilutions = h.make_attribute("dilations",config['dilation_rate'])
    a_strides = h.make_attribute("strides",config["strides"])
    a_group = h.make_attribute("group",1)
    if config["padding"] == "valid":
        padding = "VALID" 
    else:
        raise ValueError("padding not supported")
    a_auto_pad = h.make_attribute("auto_pad",padding)
    if config["data_format"] == "channels_first":
        attributes = []
    elif config["data_format"] == "channels_last":
        # 
        # data_format is defined as storage order in onnx - but not supported vor Conv https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Conv-11
        # Tensorflow focuses only on data_format = channel_last namely (NWC, NWHC) but only NCW and NCHW is supported by pytorch and onnx.
        #  
        a_storage_order = h.make_attribute("storage_order",1) 
        attributes = [a_storage_order]
    n_conv1d = h.make_node(inputs=[layer.input.name,layer.kernel.name,layer.bias.name],outputs=[layer.output.name], op_type="Conv",name=config["name"],
                            doc_string="",domain="",)
    n_conv1d.attribute.extend(attributes+[a_kernel_shape,a_strides,a_group,a_auto_pad, a_dilutions])
    w0, b0 = layer.get_weights()
    w0 = w0.T
    weight_ini = h.make_node("Constant", inputs=[], outputs=[layer.kernel.name], name=layer.kernel.name, 
        value=h.make_tensor(name=layer.kernel.name, data_type=tp.FLOAT, 
        dims=w0.shape, 
        vals=w0.flatten()))
    bias_ini = h.make_node("Constant", inputs=[], outputs=[layer.bias.name], name=layer.bias.name, 
        value=h.make_tensor(name=layer.bias.name, data_type=tp.FLOAT, 
        dims=b0.shape, 
        vals=b0.flatten()))
    return [weight_ini, bias_ini], [n_conv1d]

def make_dense_nodes(layer):
    config = layer.get_config()
    weights = layer.get_weights()
    n_dense = h.make_node(inputs=[layer.input.name,layer.kernel.name],outputs=[layer.output.name+"_MatMul"], op_type="MatMul",name=config["name"]+"MatMul",
                            doc_string="",domain="")
    n_add = h.make_node(inputs=[layer.output.name+"_MatMul",layer.bias.name],outputs=[layer.output.name], op_type="Add",name=config["name"]+"Add",
                            doc_string="",domain="")
    w0, b0 = layer.get_weights()
    weight_ini = h.make_node("Constant", inputs=[], outputs=[layer.kernel.name], name=layer.kernel.name, 
        value=h.make_tensor(name=layer.kernel.name, data_type=tp.FLOAT, 
        dims=w0.shape, 
        vals=w0.flatten()))
    bias_ini = h.make_node("Constant", inputs=[], outputs=[layer.bias.name], name=layer.bias.name, 
        value=h.make_tensor(name=layer.bias.name, data_type=tp.FLOAT, 
        dims=b0.shape, 
        vals=b0.flatten()))
    return [weight_ini,bias_ini], [n_dense, n_add]

def make_flatten_node(layer):
    config = layer.get_config()
    n_flatten = h.make_node(inputs=[layer.input.name],outputs=[layer.output.name], op_type="Flatten",name=config["name"],
                            doc_string="",domain="")
    return n_flatten

def convert_model(keras_model):
    nodes = [] 
    initializer = []
    for layer in keras_model.layers:
        if layer.__class__.__name__ == "Conv1D":
            ini, node = make_conv1d_node(layer)
            nodes +=node
            initializer += ini
        elif layer.__class__.__name__ == "Flatten":
            nodes.append(make_flatten_node(layer))
        elif layer.__class__.__name__ == "Dense":
            ini, ns = make_dense_nodes(layer)
            nodes += ns
            initializer += ini
    graph = h.make_graph(initializer+nodes, keras_model.get_config()["name"],
        [h.make_tensor_value_info(keras_model.input.name, tp.FLOAT, [1]+list(keras_model.input_shape)[1:])],
        [h.make_tensor_value_info(keras_model.output.name, tp.FLOAT, [1]+list(keras_model.output_shape)[1:])])
    onnx_model = h.make_model(graph, producer_name="My_Special_Hack") #,opset = h.make_opsetid("",11))
    #h.make_opsetid(11)
    onnx_model.opset_import[0].version = 11
    return onnx_model

def conv1d_flatten_dense():
    conv1d_model = onnx.load(f"{image_folder}/conv1d_flatten_dense.onnx")
    return conv1d_model

def do_conv1d_flatten_dense():
    onnx_model = onnx.load(f"{image_folder}/conv1d_flatten_dense.onnx")
    checker.check_model(onnx_model)

    dense=gen_dense()
    onnx_model_dense = convert_model(dense)
    checker.check_model(onnx_model_dense)
    save(onnx_model_dense, f"{temp_dir}/my_dense.onnx")
    data_input = (np.random.randint(10,size=(dense.input.shape))/10).astype(np.float32)

    session = rt.InferenceSession(f"{temp_dir}/my_dense.onnx")
    inname = [input.name for input in session.get_inputs()]
    outname = [output.name for output in session.get_outputs()]
    print("inputs name:",inname,"|| outputs name:",outname)
    data_output = session.run(outname, {inname[0]: data_input})

    print(data_output[0].shape)
    json_obj = MessageToDict(onnx_model_dense)
    json.dump(json_obj,open(f"{temp_dir}/my_dense.json","w"),indent=2,sort_keys=True)

def do_my(keras_model, name):
    onnx_model = convert_model(keras_model)
    try:
        checker.check_model(onnx_model)
    except ValidationError as ex:
        sys.stderr.writelines(f"Exception expected: {ex}\n onnx does not support the default Tensorflow/KERAS storage_order")        
    save(onnx_model, f"{temp_dir}/{name}.onnx")
    data_input = (np.random.randint(10,size=[1]+list(keras_model.input.shape[1:]))/10).astype(np.float32)
    try:
        session = rt.InferenceSession(f"{temp_dir}/{name}.onnx")
        inname = [input.name for input in session.get_inputs()]
        outname = [output.name for output in session.get_outputs()]
        print("inputs name:",inname,"|| outputs name:",outname)
        data_output = session.run(outname, {inname[0]: data_input})
        print(data_output[0].shape)
    except Exception as ex:
        sys.stderr.writelines(f"Exception expected: {ex}\n onnx does not support the default Tensorflow/KERAS storage_order\n")
        sys.stderr.writelines("onnxruntime raise 'ONNXRuntimeError'\n")        
        # should raise "ONNXRuntimeError"
    json_obj = MessageToDict(onnx_model)
    json.dump(json_obj,open(f"{temp_dir}/{name}.json","w"),indent=2,sort_keys=True)

if __name__ == "__main__":
    do_conv1d_flatten_dense()
    do_my(gen_dense(), "dense")
    do_my(gen_conv1d_flatten_dense(data_format = "channels_first"), "gen_conv1d_flatten_dense_first")
    do_my(gen_conv1d_flatten_dense(data_format = "channels_last"), "my_gen_conv1d_flatten_dense_last")
    pass
