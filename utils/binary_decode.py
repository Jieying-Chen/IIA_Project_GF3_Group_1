def crop_header(s):
    # Crop and print the header; return the cropped binary string
    occurrence = 0
    header_len = 0
    while occurrence < 2:
        if s[header_len:header_len + 8] == '00000000':
            s = s[:header_len] + '00100000' + s[header_len + 8:]     # Replace the NULL char with a space for printing
            occurrence += 1
        header_len += 8
    header_bitstring = s[:header_len]
    n = int(header_bitstring, 2)
    header_str = n.to_bytes(header_len // 8, 'big').decode()
    print("The header is \n" + header_str)

    s = s[header_len:] # Crop the header
    while s[-8:] == '00000000': # Crop the added 0 pairs
        s = s[:-8]

    return s

def to_binary(s, path):
    from bitarray import bitarray
    a = bitarray(s)
    with open(path, 'wb') as f:
        a.tofile(f)

if __name__ == "__main__":
    foo = '011001100110100101101100011001010111001100101111001101000011011100110011001110000011001000110001001100000011100100111000001100110010111001110100011010010110011001100110000000000011000100110011001100000011100100110011001100100000000001001001010010010010101000000000001100001111111'
    print(foo[-5:])