import onnx
from onnx import helper
import numpy as np
import onnxruntime as rt
import os
import torch
import sys
#1,128,12,12
N   = 1#int(sys.argv[1])
C   = 1#int(sys.argv[2])
H   = 5#int(sys.argv[3])
W   = 64#int(sys.argv[4])
###############################################################

###############################################################
###############################################################
#Softmax Layer:: _self_attn_Softmax_l= /self_attn/Add_2_output_0 -> /self_attn/Softmax_output_0
input_4d     = onnx.helper.make_tensor_value_info('/self_attn/Add_2_output_0',onnx.TensorProto.FLOAT,shape =[N,C,H,W])
_4d_Softmax        = onnx.helper.make_tensor_value_info('/self_attn/Softmax_output_0',onnx.TensorProto.FLOAT,shape =[N,C,H,W])
node_4d_softmax = onnx.helper.make_node(
    "Softmax", 
    inputs      = ["/self_attn/Add_2_output_0"],
    outputs     = ["/self_attn/Softmax_output_0"],
    axis       = -1
    )

###############################################################
#Graph input -> Softmax -> Output
graph_def = onnx.helper.make_graph(
    nodes = [node_4d_softmax],
    name = 'test-model-softmax-(-1)',
    inputs = [input_4d],
    outputs = [_4d_Softmax]
    )
# Create the model (ModelProto)
model_def = onnx.helper.make_model(graph_def, producer_name='kc')
model_def.opset_import[0].version = 13
onnx.checker.check_model(model_def)
#write model and image in a path
onnx.save(model_def,"deploy.onnx")