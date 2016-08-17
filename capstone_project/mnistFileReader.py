import struct
import os
import numpy as np

files = ["t10k-labels-idx1-ubyte",
         "train-labels-idx1-ubyte",
         "t10k-images-idx3-ubyte",
         "train-images-idx3-ubyte"]


def loadfile(file_name):
    """WOP. currently:
    Gets a file_name and print the first 16 bytes as 4 32-bit integers using both methods of 'numpy.fromfile' and 'struct.unpack'."""

    count = 4
    with open(file_name, 'rb') as file:
        print np.fromfile(file, dtype='>i4', count=count)

    with open(file_name, 'rb') as file:
        magic_number = struct.unpack(">IIII", file.read(count * 4))
        print magic_number


for file in files:
    print "%s file has a size of %i bytes" % (file, os.path.getsize(file))
    loadfile(file)
    print
