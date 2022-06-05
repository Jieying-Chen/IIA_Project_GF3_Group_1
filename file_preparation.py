from bitarray import bitarray 
from bitarray.util import int2ba, ba2int
import numpy as np
from utils import binary_decode
import os


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
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, 'plugfest')

    file_size = decoded[0:32]
    sizeba = bitarray(list(file_size), endian='little')
    size = ba2int(sizeba)

    first_few_words_ba = bitarray(list(decoded[32:800]))
    first_few_words = first_few_words_ba.tobytes().decode('ascii', errors='ignore')
    print(first_few_words)

    file_type = str(input("Enter file type: "))
    file_name_size = int(input("Enter file name length: "))

    if file_type == 'txt':
        #decoded = decode.qpsk_decode(deconvolved)
        output_filename = "text.txt"
        n_range = int(8 * (len(decoded)//8))
        x = bitarray(list(decoded[32:n_range]))
        s = x.tobytes().decode('ascii', errors='ignore')

        s = str(size) + ' ' + s

        f = open(os.path.join(output_dir, output_filename), mode='w')
        #f = open(output_filename, mode='w')
        print(s)
        f.write(s)
        f.close()
        return
    elif file_type == 'tif' or file_type == 'wav':
        bstring = ''.join(decoded[8*(file_name_size+1)+32:])
        bin_file_name = "result.bin"
        binary_decode.to_binary(bstring, os.path.join(output_dir, bin_file_name))
        return
    else:
        return decoded[8*(file_name_size+1)+32:]




if __name__ == "__main__":

    encode = file_encode()
    cwd = os.getcwd()
    output_dir = os.path.join(cwd, 'plugfest')

    file_size = encode[0:32]
    sizeba = bitarray(list(file_size), endian='little')
    size = ba2int(sizeba)

    first_few_words_ba = bitarray(list(encode[32:160]))
    first_few_words = first_few_words_ba.tobytes().decode('ascii', errors='ignore')
    print(first_few_words)
    decoded = encode
    file_type = str(input("Enter file type: "))
    file_name_size = int(input("Enter file name length: "))
    output_filename = "text.txt"
    n_range = int(8 * (len(decoded)//8))
    x = bitarray(list(decoded[32:n_range]))
    s = x.tobytes().decode('ascii', errors='ignore')
    s = str(size) + ' ' + s
    f = open(os.path.join(output_dir, output_filename), mode='w')
    f.write(s)
    f.close()