
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
import random as rd
from collections import OrderedDict
from struct import pack, unpack
import bitstring
import csv
from random import seed


pos_list=[] # Storing the bit flip position
bit_width = 32 # Data type used is single precision
faulty_entry_list=[]
new_list=[]


def bitFlip(num, pos):
    '''
        Flip the bit position of the number in the binary representation
        The argument 'pos' starts from 0 to 31 MSB-->LSC
    '''
    x=bitstring.BitArray(float=num, length=32)
    str=x.bin # Converting to string in binary format: 11000011...
    loc=31-pos
# We need to revesrse the order of the bit to be flipped because the difference of
# the pos given from the transLayer is counted from right to left. The bit location,
# however, to be flipped must be counted from left to right (standard Floating-point number)
    if str[loc]=="1":
        if loc==0:
            new="".join(("0",str[1:]))
        elif loc==len(str):
            new="".join((str[:loc],"0"))
        else :
            new="".join((str[:loc],"0",str[loc+1:]))
    elif str[loc]=="0":
        if loc==0:
            new="".join(("1",str[1:]))
        elif loc==len(str):
            new="".join((str[:loc],"1"))
        else :
            new="".join((str[:loc],"1",str[loc+1:]))
    #print(new) # binary string
    str_int=int(new,2)
    byte=(str_int).to_bytes(4, byteorder='big')
    f=unpack('>f', byte)[0]
    pos_list.append(pos) # Keep adding the bit positions to the 'global list'

    return f


def FaultInjection(fault_rate, data):
    '''
        Injecting faults to the 1D array with a specific fault rate
        It will return the mutated data
    '''
    total_bits = int(bit_width*data.size) # Counting the total number of data bits    
    num_faulty_bits = int(round(total_bits*fault_rate)) # Counting the number of faulty bits # the whole network
    # The line of code below generates the list of bit position to be flipped
    bit_list=rd.sample(range(1, total_bits), num_faulty_bits)
    for i in range(len(bit_list)):
        faulty_entry_list.append(int(float(bit_list[i])/bit_width))
        [q,r] = divmod(bit_list[i], 32)
        #print('[q, r]:', q,r)
        if q==0 and r==0:
            data[faulty_entry_list[i]] = bitFlip(data[faulty_entry_list[i]], 31)
        if r==0:
            data[faulty_entry_list[i]] = bitFlip(data[faulty_entry_list[i]-1], 0)
        else:
            data[faulty_entry_list[i]] = bitFlip(data[faulty_entry_list[i]], 32-r)
    print('|| Number of faulty bits:', num_faulty_bits)
    #print('|| faulty_entry_list:', faulty_entry_list)
    #print('|| bit_list: ', bit_list)
    #print('|| Bit list generated as:', bit_list) # Checkpoint
    return data # Return mutated data

def transformLayer(fault_rate, params, layers_name, filename, saved_file, layer):
    '''
        Pass the name of the layers to the method, mutated it, store it back to the layer
        For example, layer='conv1' --> Concatenate weight and bias to one
        layer='fc1'--> do the same task
        Call the functions FaultInjection, bitFlip
    '''
    print('============================================================')
    print('     Transforming specified layer of the network')
    print('============================================================')
    print('|| Filename of the fault-free weights : ', filename,)
    print('|| Fault rate :                         ', fault_rate)
    print('|| Filename of the mutated weight :     ', saved_file)
    print('|| Faults are injected into the following layer:')
    join_params=[] # Concatenated array of the weight and bias array
    if layer.find('conv') != -1:
        weight=layer+'.weight' # weight='conv1.weight' for indexing the params
        bias=layer+'.bias'
        # layer.find('text') return the index where the text begins in the string
        # It returns -1 if the text is not found in the string
        print('|| ',layer)
        mutated_params=[]
        kernels, depth, row, col = params[layers_name.index(weight)].numpy().shape
        row_b = params[layers_name.index(bias)].numpy().shape
        size_weight=params[layers_name.index(weight)].numpy().size
        size_bias=params[layers_name.index(bias)].numpy().size
        # Advanced technique used to nest dimensions in the reshaping will be demonstrated in the function "transformNetwork"
        weight_params=params[layers_name.index(weight)].numpy().flatten()
        bias_params=params[layers_name.index(bias)].numpy().flatten()
        join_params=np.concatenate([weight_params, bias_params],0)
        mutated_params=FaultInjection(fault_rate, join_params)
        #Reshaping
        params[layers_name.index(weight)]=torch.from_numpy(mutated_params[:size_weight].reshape(kernels, depth, row, col))
        params[layers_name.index(bias)]=torch.from_numpy(mutated_params[size_weight:size_bias+size_weight].reshape(row_b))

    elif layer.find('fc') != -1:
        weight=layer+'.weight'
        bias=layer+'.bias'
        print('|| ',layer)
        mutated_params=[]
        row, col = params[layers_name.index(weight)].numpy().shape
        row_b = params[layers_name.index(bias)].numpy().shape # Row of the bias
        size_weight=params[layers_name.index(weight)].numpy().size
        size_bias=params[layers_name.index(bias)].numpy().size

        weight_params=params[layers_name.index(weight)].numpy().flatten()
        bias_params=params[layers_name.index(bias)].numpy().flatten()

        join_params=np.concatenate([weight_params, bias_params],0)
        mutated_params=FaultInjection(fault_rate, join_params)

        params[layers_name.index(weight)]=torch.from_numpy(mutated_params[:size_weight].reshape(row, col))
        params[layers_name.index(bias)]=torch.from_numpy(mutated_params[size_weight:size_bias+size_weight].reshape(row_b))


    print('|| Mutated weights saved ||')
    new_data_list  = dict(zip(layers_name, params))
    new_data_odict = OrderedDict(new_data_list.items()) # Converting to orderedict
    torch.save(new_data_odict, saved_file)
    print('============================================================')



def transformNetwork(fault_rate, params, layers_name, filename, saved_file):
    '''
    params and layers_name are list with torch tensor
    In the lenet_inferencing.py, the data has been loaded by torch.load(filename, map_location='cpu')
    so that all tensors are moved to CPU memory afterward. That is why we do not need to include cpu() in the code line:
    join_params=np.concatenate([join_params, params[i].numpy().flatten()], 0)
    '''
    print('===========================Start============================')
    print('     Transforming all layers of the network')
    print('============================================================')
    print('|| Filename of the fault-free weights : ', filename,)
    print('|| Fault rate :                         ', fault_rate,)
    print('|| Filename of the mutated weights :     ', saved_file,)
    print('|| Faults are injected into all layers:')
    #print(layers_name)

    mutated_params=[] # Mutated data will be stored here
    join_params=[] # Unified array of params of all layers
    size_list=[] # Size of the layers
    x=0 # temporary variable to compute the range of array slicing
    #new_list=[] # Store the indexing range for array slicing
    for i in range(len(layers_name)):
        join_params=np.concatenate([join_params, params[i].numpy().flatten()], 0)
        size_list.append(params[i].numpy().flatten().size)
    #print('original size_list :', size_list)
    #print('the size of size_list ', len(size_list))
    size_list=np.flip(np.append(np.flip(size_list),[0]))
    ##print('sorted and zero-added size_list :', size_list)
    #print('the size of size_list ', len(size_list))
    # np.flip(a) reverses the order of the numpy array
    # We add 0 to the first position of the current size list for the upcoming mapping algorithm
    #print('|| size_list:', size_list)
    mutated_params=FaultInjection(fault_rate, join_params)

    # Reshaping the params
    for i in range(len(size_list)):
        x+=size_list[i]
        new_list.append(x)
    for i in range(len(layers_name)-1):
        params[i]=torch.from_numpy(mutated_params[new_list[i]:new_list[i+1]].reshape(params[i].numpy().shape))
        # Reshaping automatically without temporarily storing the shape of the original data array
    #print('|| new_list:', new_list)
    print('|| Mutated weights saved ||')
    new_data_list  = dict(zip(layers_name, params))
    new_data_odict = OrderedDict(new_data_list.items()) # Converting to orderedict
    torch.save(new_data_odict, saved_file)
    print('============================================================')

def layerInfo(local_faulty_entry_list, layers_name, local_new_list, local_pos_list):
    '''
    local_faulty_entry_list: list of faulty entry, e.g [40206, 50882, 4140, 26270]
    layers_name : Name of the layer, e.g. conv1.weight, ....
    local_new_list: sorted and accummulated index of each layer, e.g [0, 150, 156, 2556, 2572, 50572, 50692, 60772, 60856, 61696, 61706]
    local_pos_list: list of bit flip in the faulty entries, e.g [13, 24, 26, 16]
    This function aims to return the number of sign bit and e7 bit flip in each layer.
    For example:
    sign bit flip: conv1.weight, fc1.weight
    e7 bit flip: conv2.weight
    flip at the exponent part and sign part: conv1.weight, fc1.weight,conv2.weight
    The algorithm can be described as follows:
    - Step 1: create a dict of layer_names and local_new_list, e.g. {'conv1.weight':[0, 150], 'conv1.bias': [150, 156]}
    Create three lists of e7 bit flip, sign bit flip, and exponent+sign part flip
    - Step 2: identify if the e7 bits were flipped and map the index to the` fault faulty_entry_list
    - step 3: check if the faulty-entry-list in the values range of each dict key and return the name of the key
    - step 4: append the founded key to the e7 list
    - step 5: do the same for sign bit flip and the rest
    - step 6: Print out three list.
    '''
    temp=[]
    e7=[]
    sign=[]
    exp_sign=[]
    for i in range(len(local_new_list)-1):
        temp.append([local_new_list[i], local_new_list[i+1]])
    dict1=dict(zip(layers_name, temp))
    layerID=list(dict1.keys())
    param=list(dict1.values())
    for i in range(len(local_pos_list)): # Walk through the pos list of bit flip
        if local_pos_list[i]==31:
            for j in range(len(dict1)):
                if local_faulty_entry_list[i]>= param[j][0] and local_faulty_entry_list[i]<=param[j][1]:
                    sign.append(layerID[j])
        if local_pos_list[i]==30:
            for j in range(len(dict1)):
                if local_faulty_entry_list[i]>= param[j][0] and local_faulty_entry_list[i]<=param[j][1]:
                    e7.append(layerID[j])
        if local_pos_list[i]>22:
            for j in range(len(dict1)):
                if local_faulty_entry_list[i]>= param[j][0] and local_faulty_entry_list[i]<=param[j][1]:
                    exp_sign.append(layerID[j])
    print('\ne7:', e7)
    print('\nsign:', sign)
    print('\nexp_sign', exp_sign)


def stats():
    sign=0
    e7=0
    x=0
    x=sum(1 for item in pos_list if item>22)
    sign=sum(1 for item in pos_list if item==31)
    e7=sum(1 for item in pos_list if item==30)
    print("\n|| bit flip at exponent or sign part of the number: {}/{} ({:.2f} %)".format(x,len(pos_list), 100.0*x/len(pos_list)))
    print("|| bit flip at sign part of the number: {}/{} ({:.2f} %)".format(sign,len(pos_list), 100.0*sign/len(pos_list)))
    print("|| bit flip at e7 of the number: {}/{} ({:.2f} %)".format(e7,len(pos_list), 100.0*e7/len(pos_list)))
    print('===========================================================')

def main():
    parser = argparse.ArgumentParser(description='Fault injection experiment')
    parser.add_argument('--loaded-file', type=str, default='mnist_cnn.pt',
                        help='File name to be loaded for being manipulated')
    parser.add_argument('--saved-file', type=str, default='default_fault.pt',
                        help='File name to be saved')
    parser.add_argument('--fault-rate', type=float, default=0.00001,
                        help='Fault rate')
    parser.add_argument('--trans-network', action='store_true',
                        help='Transform all the layers')
    parser.add_argument('--trans-layer', action='store_true',
                        help='Transform a specified layer')
    parser.add_argument('--show-stats', action='store_true',
                            help='Showing statistics of the fault injection')
    parser.add_argument('--show-layer-info', action='store_true',
                        help='Showing statistics of layers to be mutated')
    args = parser.parse_args()

    fn = args.loaded_file # Remember to cast the new model here.
    fault_rate=args.fault_rate

    data = torch.load(fn, map_location='cpu')
    #data = torch.load(fn)
    layers_name = list(data.keys())
    params = list(data.values())
    # Load the file fn with the extension '.pt'
    # Store the name of the layers, e.g. fc1.weight, to the list layers_name
    # Store the parameters in 4D array to the list params

    #layer = 'fc1.weight' # layer to be injected faults
    #layer= 'conv2.weight'
    #layer='conv2'
    layers_list=['conv1',
        'conv2',
        'fc1',
        'fc2',
        'fc3'
                ]
    if args.trans_layer:
        transformLayer(fault_rate, params, layers_name, fn, args.saved_file, layers_list[4])
        if args.show_stats:
            stats()
        if args.show_layer_info:
            layerInfo(faulty_entry_list, layers_name, new_list, pos_list)

    if args.trans_network:
        transformNetwork(fault_rate, params, layers_name, fn, args.saved_file)
        if args.show_stats:
            stats()
        if args.show_layer_info:
            layerInfo(faulty_entry_list, layers_name, new_list, pos_list)



if __name__ == "__main__":
    seed(2)
    main()
