from bitarray import bitarray 
from bitarray.util import int2ba, ba2int
import numpy as np


def file_encode(filesize = 114514, filename = 'file', txt_name = 'file_test.txt'):
    ''' encode file into binary string'''

    size = bitarray()
    size = int2ba(filesize,length=32,endian = 'little')
    size_list = size.tolist()
    #print(size_list)

    filename = filename + '\0'
    name = bitarray()
    name.frombytes(filename.encode('ascii'))
    name_list = name.tolist()
    #print(name_list)

    text = bitarray()
    raw_data = open(txt_name, "r").read()
    text.frombytes(raw_data.encode('ascii'))
    text_list = text.tolist()
    #print(text_list)

    data_list = np.array(size_list+name_list+text_list)
    return data_list



def file_decode(decoded):
    '''decode binary string using ascii'''

    file_size = decoded[0:32]

    sizeba = bitarray(list(file_size), endian='little')
    size = ba2int(sizeba)

    #decoded = decode.qpsk_decode(deconvolved)
    output_filename = "text.txt"
    n_range = int(8 * (len(decoded)//8))
    x = bitarray(list(decoded[32:n_range]))
    s = x.tobytes().decode('ascii', errors='ignore')

    s = str(size) + ' ' + s
    f = open(output_filename, mode='w')
    f.write(s)
    f.close()


if __name__ == "__main__":

    encode = file_encode()
    file_decode(encode)