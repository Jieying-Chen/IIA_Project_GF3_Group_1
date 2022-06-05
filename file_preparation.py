from bitarray import bitarray
from bitarray.util import int2ba
import numpy as np


#encode
filesize = 114514
size = bitarray()
size = int2ba(filesize,endian = 'little')
buffer = size.tolist()
size_list = [0] * 32
size_list[:len(buffer)] = buffer
#print(size_list)

filename = 'filename\0'
name = bitarray()
name.frombytes(filename.encode('ascii'))
name_list = name.tolist()
#print(name_list)

text = bitarray()
raw_data = open("file_test.txt", "r").read()
text.frombytes(raw_data.encode('ascii'))
text_list = text.tolist()
#print(text_list)

data_list = np.array(size_list+name_list+text_list)
print(data_list)





#decode
#decoded = decode.qpsk_decode(deconvolved)
decoded = data_list
output_filename = "text.txt"
n_range = int(8 * (len(decoded)//8))
x = bitarray(list(decoded[:n_range]))
s = x.tobytes().decode('ascii', errors='ignore')

f = open(output_filename, mode='w')
f.write(s)
f.close()
