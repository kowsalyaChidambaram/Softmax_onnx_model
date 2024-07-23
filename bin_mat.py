import struct
import sys
import numpy as np
import torch
import os
#import tensorflow as tf

bin_ = str(sys.argv[1])
N = int(sys.argv[2])
C = int(sys.argv[3])
H = int(sys.argv[4])
W = int(sys.argv[5])

bin_file = bin_+ '.bin'

txt_file = bin_file + '_text.txt'
with open(bin_file, "rb") as f:
    with open((txt_file),'w') as file:
       byte = f.read(4)
       while byte != b"":
          file.write(str(struct.unpack('f', byte)[0]) + '\n')
          byte = f.read(4)
    file.close()
f.close()

list_of_lists = []
final = []
text_file = open(txt_file, "r")
lines = text_file.read().split(',')
with open(txt_file) as f:
    for line in f:
      for x in line.split(','):
        list_of_lists.append(x.strip())
        
print(len(list_of_lists))
list_len = 0
for i in list_of_lists:
  if i != "" and list_len < 320:
    final.append(float(i))
    list_len = list_len + 1
final = np.asfarray(final)
final = final.reshape((N,C,H,W))
final = torch.tensor(final)
final = final.type(torch.FloatTensor)
final = final.numpy()
np.save(bin_+'.npy',final)
#print(final)
def create_mat_file(file_name,data_order,number_of_inputs,height,width,channels,data_type,scale_factor,buffer_np):
    if data_order == 5:
        mat_header_part1 = np.array([data_order,number_of_inputs,channels,height,width,data_type],dtype = np.uint32)
    else:
        print("Unsupported data order/n")
    mat_header_part2 = np.array([scale_factor],dtype = np.float32)
    with open(file_name,"w") as file_handle:
        mat_header_part1.tofile(file_handle)
        mat_header_part2.tofile(file_handle)
        buffer_np.tofile(file_handle)
input_file = bin_+'.npy'
npy_file = np.load(input_file)
npy_file = np.transpose(npy_file, [0,1,2,3])
out_name = bin_+'.mat'
number_of_inputs = npy_file.shape[0]
channels = npy_file.shape[1]
height =  npy_file.shape[2]
width =  npy_file.shape[3]
print(number_of_inputs)
print(channels)
print(height)
print(width)
data_order = 5
data_type = 64#
scale_factor = 0.0
create_mat_file(out_name,data_order,number_of_inputs,height,width,channels,data_type,scale_factor,npy_file)
#os.remove(bin_+'.npy')
#os.remove(bin_file + '_text.txt')