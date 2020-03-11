'''
    This Python script is used to injecting faults to the DNN model.
    It is explicitly used for single-precision data
    Input:
        -- Parameters files loading from the torch.load('file_name.pt', map_location=device)
        -- Types of fault injection: layer-wise fault injection, random network
    Output:
        -- Set of mutated parameters
    Functionality of each function:
        -- bitFlip: Perform bit flips at specified position (pos) of a number (num)
        -- fault_injection: Perform mutating given data (data) with the fault rate (fault_rate) with the fault pattern (seed)
            Notice that pseudo random number generator is used with the seed value
        -- randNetwork: Perform bit flips accross all parameters of the network randomly
        -- populationNetwork: Perform bit flips at all layers of the network with the same fault rate
        -- layer: Perform layer-wise fault injection to the layer with the name passed from the argument of the main file
            e.g., python alexnet.py --layer-name conv1.weight

'''


import numpy as np
import bitstring
import random as rd
from struct import pack, unpack
import torch

pos_list=[] # Storing the bit flip position
qbit = 32 # single-precision data
def bitFlip(num, pos):
    # qbit = 32 # 32 bit floating point
    x=bitstring.BitArray(float=num, length=qbit)
    str=x.bin # Converting to string in binary format: 11000011...
    # print("Original value:", num)
    # print("Binary representation: ",str)
    # print("Bit location to be flipped counting from left to right:", pos) # Pos starts from 0 unitl 7
    if str[pos]=="1":
        if pos==0:
            new="".join(("0",str[1:]))
        elif pos==len(str):
            new="".join((str[:pos],"0"))
        else :
            new="".join((str[:pos],"0",str[pos+1:]))
    elif str[pos]=="0":
        if pos==0:
            new="".join(("1",str[1:]))
        elif pos==len(str):
            new="".join((str[:pos],"1"))
        else :
            new="".join((str[:pos],"1",str[pos+1:]))
    str_int=int(new,2)
    byte=(str_int).to_bytes(4, byteorder='big')
    f=unpack('>f', byte)[0]
    pos_list.append(pos) # Keep adding the bit positions to the 'global list'
    return f

def fault_injection(fault_rate, data, seed):
    rd.seed(seed)
    total_bits = int(32*data.size) # Counting the total number of data bits
    num_faulty_bits = int(round(total_bits*fault_rate)) # Counting the number of faulty bits
    bit_list=rd.sample(range(1, total_bits), num_faulty_bits)    # The line of code below generates the list of bit position to be flipped
    print("|| Fault injection started")
    print("|| SEED=", seed)
    print('\n=================== FAULT INJECTION STATS =================')
    print('|| Number of faulty bits:', num_faulty_bits)
    faulty_entry_list=[] # We have to put this variable here because it can be overloaded in case of population based injection
    for i in range(len(bit_list)):
        faulty_entry_list.append(int(float(bit_list[i])/qbit))
        [q,r] = divmod(bit_list[i], qbit)
        if q==0 and r==0:
            data[faulty_entry_list[i]] = bitFlip(data[faulty_entry_list[i]], 0)
        if r==0:
            data[faulty_entry_list[i]] = bitFlip(data[faulty_entry_list[i]-1], qbit-1)
        else:
            data[faulty_entry_list[i]] = bitFlip(data[faulty_entry_list[i]], r-1)
    stats()
    return data # Return mutated datadef bitFlip(num, pos):

def stats():
    sign=0
    e7=0
    x=0
    x=sum(1 for item in pos_list if item>22)
    sign=sum(1 for item in pos_list if item==31)
    e7=sum(1 for item in pos_list if item==30)
    print("|| bit flips at exponent or sign location of the number: {}/{} ({:.2f} %)".format(x,len(pos_list), 100.0*x/len(pos_list)))
    print("|| bit flips at sign location of the number: {}/{} ({:.2f} %)".format(sign,len(pos_list), 100.0*sign/len(pos_list)))
    print("|| bit flips at e7 location of the number: {}/{} ({:.2f} %)".format(e7,len(pos_list), 100.0*e7/len(pos_list)))
    print('===========================================================')

def randNetwork(args, fault_rate, model, seed):
    layers_name=list(model.keys())
    params=list(model.values())
    index_list=[]#1D array storing the index of original params
    size_list=[] #1D array storing the size of each params
    join_params=np.array([], dtype='float32') #1D array storing the values of parameters including weights and biases
    mutated_params=[] #1D array storing the mutated values from the join_params
    slicing_list=[0] # 1D value storing the slicing index: [0 5 5+4 ...]
    x=0
    key_list=[] #1D array storing the name of layers
    for i in range(len(params)):
        if layers_name[i].find('weight')!=-1 and (not layers_name[i].find('bn')!=-1) :
            index_list.append(i)
            size_list.append(len(params[i].numpy().flatten()))
            join_params=np.concatenate([join_params, params[i].numpy().flatten()],0)
            key_list.append(layers_name[i])

    for i in range(len(size_list)):
        x+=size_list[i]
        slicing_list.append(x)
    mutated_params=fault_injection(fault_rate, join_params, seed)
    for i in range(len(slicing_list)-1):         #Assigning new parameters to the corresponding layers
        model[key_list[i]]=torch.from_numpy(mutated_params[slicing_list[i]:slicing_list[i+1]].astype(np.float32).reshape(params[index_list[i]].numpy().shape))
    return model


def layer(args, fault_rate, model, seed, layer):
    '''
        Injecting faults to a specific layer
    '''
    # layers_name=list(model.keys())
    # params=list(model.values())
    # layer=args.layer_name
    total_bits = int(qbit*model[layer].numpy().flatten().size) # Counting the total number of data bits
    num_faulty_bits = int(round(total_bits*fault_rate)) # Counting the number of faulty bits
    if num_faulty_bits>0:
        print('Number of faulty bits:', num_faulty_bits)
        print("|| Layer to be injected:", layer,"\n")
        mutated_params = fault_injection(fault_rate, model[layer].numpy().flatten(), seed)
        model[layer] = torch.from_numpy(mutated_params.reshape(model[layer].numpy().shape))
        return model
    else:
        print("\n ***TRY AGAIN WITH THE HIGHER FAULT RATE***\n")
        raise ValueError
