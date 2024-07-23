import onnx
from onnx import helper
import onnxruntime as rt
import numpy as np
import struct 
import os 
import sys
import cv2
from csv import reader
def parse_mat(mat_path):
    with open(mat_path, "rb") as f:
        data_order = float(struct.unpack('I', f.read(4))[0])
        number_of_inputs = float(struct.unpack('I', f.read(4))[0])
        channels = float(struct.unpack('I', f.read(4))[0])
        height = float(struct.unpack('I', f.read(4))[0])
        width = float(struct.unpack('I', f.read(4))[0])
        data_type = float(struct.unpack('I', f.read(4))[0])
        scale_factor = float(struct.unpack('f', f.read(4))[0])
        cellValue = 0
        img = np.arange(number_of_inputs* channels* height* width, dtype = float)
        #data_order, #number_of_inputs, #channels, height, #width, #data_type, #scale_factor
        byte = f.read(4)
        while byte != b"":
            img[cellValue] = float(struct.unpack('f', byte)[0])
            byte = f.read(4)
            cellValue = cellValue +1
        f.close()
    return img
def add_out(ONNX_PATH):
    #load onnx model and add output, input tensors to the list
    model = onnx.load(ONNX_PATH)
    intermediate_layer_value_info = helper.ValueInfoProto()

    out = []
    inp=[]
    for j in range(len(model.graph.output)):
       out.append(model.graph.output[j].name)

    for i in range(len(model.graph.input)):
       inp.append(model.graph.input[i].name)
       intermediate_layer_value_info.name = model.graph.input[i].name
       model.graph.output.append(intermediate_layer_value_info)
    for i in range(len(model.graph.node)):
        for j in range (len(model.graph.node[i].output)):
            name = model.graph.node[i].output[j]
            intermediate_layer_value_info.name = model.graph.node[i].output[0]
            model.graph.output.append(intermediate_layer_value_info)
    onnx.save(model, Model_out_path)
# Please make sure below values are aligned to model inputs
ONNX_PATH = str(sys.argv[1])
INPUT = str(sys.argv[2])
BATCH = int(sys.argv[3])
CHANNEL = int(sys.argv[4])
HEIGHT = int(sys.argv[5])
WIDTH = int(sys.argv[6])

MODEL_PATH = ONNX_PATH.replace("deploy.onnx","")
Model_out_path = MODEL_PATH+"deploy_out_model.onnx"
MODEL_PATH = MODEL_PATH+"dumps-model/"


#load onnx model and add output, input tensors to the list
add_out(ONNX_PATH)

if(INPUT.endswith(".mat")):
    #parse the header part
   img = parse_mat(INPUT)
        
X_test = img.reshape((BATCH,CHANNEL,HEIGHT, WIDTH))

#open the onnx model with in/out tensors added
# Load the model and append all output names
model = onnx.load(Model_out_path)
listoutput = []
for i in range(len(model.graph.output)):
   listoutput.append(model.graph.output[i].name)

# Run
sess = rt.InferenceSession(Model_out_path)
input_name = sess.get_inputs()[0].name
pred = sess.run(listoutput, {input_name: X_test.astype(np.float32)})

# Change the below string to dumps-{modelname}
if not os.path.isdir(MODEL_PATH):
   os.makedirs(MODEL_PATH)
   
for i in range(len(listoutput)):
   print("Dumping ", listoutput[i], " shape ", pred[int(i)].shape)
   listoutput[i]=listoutput[i].replace('/','_')
   listoutput[i]=listoutput[i].replace(':','_')
   #listoutput[i]=listoutput[i]+ '_' + str(i)
   with open(MODEL_PATH+'{}'.format(listoutput[i]+".bin"), "wb") as f:
      for val in pred[i].flatten():
         f.write(struct.pack('f', val))
os.remove(Model_out_path)
