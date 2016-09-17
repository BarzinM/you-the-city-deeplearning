import os
import numpy as np
import cv2


class SVHNParserError(Exception):
    pass


def decompress(file_name):
    """
    Untars a file with '.tar.gz' extension.
    Example: decompress('somefile.tar.gz') -> decompresses somefile.gz

    Inputs: TODO
    """
    import tarfile
    tar = tarfile.open(file_name)
    tar.extractall()
    tar.close()


def download(URL):
    """
    Downloads a file from the `URL`.

    Inputs:
    - URL: complete address of the file.
    """

    import urllib.request
    urllib.request.urlretrieve(URL, os.path.basename(URL))


def maybeDownload(file_name, force=False):
    """
    Checks to see if a file from MNIST exists or not. if not, it is downloaded.

    Inputs:
    - file_name: name of the file.
    """
    def addressExists(address):
        return os.path.isfile(address) or os.path.isdir(address)
    # see if file exists
    if addressExists(file_name) and not force:
        print(file_name, 'already exists')
        return  # if it exists then return.

    # see if the compressed version of the file (.gz extension) exists.
    compressed_file = file_name + '.tar.gz'
    if addressExists(compressed_file) and not force:
        print('Found %s. Decompressing ...' % compressed_file, end=' ')

        # if exists then decompress it.
        decompress(compressed_file)
        print('Finished')
        return  # and return.

    # if none of the above, then download the compressed file.
    print('Downloading', file_name)
    download('http://ufldl.stanford.edu/housenumbers/' + compressed_file)
    print("%s file has a size of %i bytes" %
          (compressed_file, os.path.getsize(compressed_file)))

    # and decompress it.
    decompress(compressed_file)
    print('Downloaded and decompressed', file_name)


def getNumberOfFiles(mat_file):
    import h5py
    f = h5py.File(mat_file, 'r')
    bbox_dataset = f["digitStruct"]["bbox"]
    number_of_files = bbox_dataset.shape[0]
    return number_of_files


def getLabels(mat_file, indieces=None):
    import h5py
    f = h5py.File(mat_file, 'r')
    names_dataset = f["digitStruct"]["name"]
    bbox_dataset = f["digitStruct"]["bbox"]
    number_of_files = bbox_dataset.shape[0]

    def _getLabels(index):
        name = f[names_dataset[index][0]]
        file_name = ''.join(chr(x) for x in name)
        property_label = f[bbox_dataset[index][0]]["label"]
        labels = []
        for label in property_label:
            try:
                labels.append(f[label[0]][0][0])
            except AttributeError:
                labels = [property_label[0, 0]]
        return file_name, labels

    if indieces is None:
        names_list = []
        labels_list = []
        for index in range(number_of_files):
            name, labels = _getLabels(index)
            names_list.append(name)
            labels_list.append(labels)
        return names_list, labels_list

    elif type(indieces) is int:
        return _getLabels(indieces)

    elif type(indieces) in (list, np.ndarray):
        names_list = []
        labels_list = []
        for index in indieces:
            name, labels = _getLabels(index)
            names_list.append(name)
            labels_list.append(labels)
        return names_list, labels_list
    else:
        raise SVHNParserError(
            "The type of `indieces` argument should be `int` or `list`. The argument can also be left unassigned to parse all the files. The type received was %s." % type(indieces))

def parseLabel(label, maximum_number_of_digits=2):
    number_of_digits = len(label)
    if number_of_digits>maximum_number_of_digits:
        number_of_digits = 0.0
    parsed = [float(number_of_digits)]+[0.0]*maximum_number_of_digits
    parsed[1:len(label)+1]=label[:maximum_number_of_digits]
    # for digit in range(len(label)):
    #     parsed[digit+1] = label[digit]
    return parsed

def getImage(file_name, directory='.', shape=None):
    def readImageFile(address):
        im = cv2.imread(address)
        return im
    if type(file_name) is list:  # make sure if this  works for list of strings
        if shape is None:
            raise SVHNParserError(
                "If `file_name` is a list of files, the `shape` argument should be assigned a 2 element array (eg. shape=(width, height)).")
        number_of_images = len(file_name)
        width, height = shape
        data = np.empty([number_of_images, height, width, 3])
        for i in range(number_of_images):
            image = readImageFile(directory+ '/' + file_name[i])
            image_height,image_width=image.shape[:2]
            if image_height<height and image_width<width:
                position_0 = np.random.randint(height - image_height)
                position_1 = np.random.randint(width - image_width)
                data[i,position_0:image_height + position_0,
                           position_1:image_width + position_1] = image
            else:
                data[i] = cv2.resize(image, (width, height))
        return data
    elif type(file_name) is str:
        return readImageFile(directory+ '/' + file_name)
    else:
        raise SVHNParserError(
            "The argument `file_name` should be either a file name (`str`) or a list of file names (`list`). The type of the argument received was %s" % type(file_name))


def toOnehot(array, num_classes=10):
    """
    Takes a 1 dimensional array and returns a 2 dimensional array of one-hots with the dimension 0 with the same size of input array and dimension 1 with the size of `num_of_classes`.

    Inputs:
    - array: 1D array in which each element is a class index.
    - num_classes: number of classes that each element of `array` falls into.

    Outputs:
    - A 2D array of one-hots.
    """
    count = len(array)
    onehot_array = np.zeros((count, num_classes), np.int8)
    onehot_array[np.arange(count), array] = 1
    return onehot_array


def parseMnistFile(file_name):
    """
    Gets a file_name belonging to MNIST. If the file contains labels then the function returns an 1D array of labels. If the file contains images then a 3D array is returned with the size of [number of images, 28, 28]. For more informatin: http://yann.lecun.com/exdb/mnist/

    Inputs:
    - file_name: name of the file including the extension (e.g. 't10k-labels-idx1-ubyte.gz').
    """

    with open(file_name, 'rb') as file:
        magic_number, count = np.fromfile(file, dtype='>u4', count=2)
        if magic_number == 2049:
            print(file_name, 'is a lable file')
            array = np.fromfile(file, dtype=np.uint8)
            if len(array) != count:
                raise ValueError
        if magic_number == 2051:
            print(file_name, 'is a data file')
            num_rows, num_columns = np.fromfile(file, dtype='>u4', count=2)
            if num_rows != 28 or num_columns != 28:
                raise ValueError
            array = np.fromfile(file, dtype=np.uint8).astype(float)
            if len(array) != count * 28 * 28:
                raise ValueError
            array = array.reshape((count, 28, 28))
        return array


# def seperateDataset(data, labels):
#     """
#     Separates data based on labels into a dictionary. Each dictionary key is one class and the value of that key is an array of all the data with the same class.
#     Inputs:
#     - data: an array of data.
#     - labels: an array of labels corresponding to the data array.

#     Output:
#     - A dictionary in which keys are the unique classes and values are arrays of data associated with the class represented by key.
#     """
#     uniques, index = np.unique(labels, return_inverse=True)
#     dictionary = dict.fromkeys(uniques)
#     for unique in uniques:
#         array = data[np.where(labels == unique)]
#         dictionary[unique] = array
#     return dictionary


def maybePickle(array, file_name, force=False):
    """
    Pickles an array into a file if it doesn't already exists.

    Inputs:
    - array: a numpy array.
    - file_name: name of the file without an extension (e.g. maybePickle(my_array, 'somefile')) to save the pickled array into `somefile.npy`.
    - force [optional]: force recreating the file even if it already exists.
    """
    if os.path.isfile(file_name + '.npy') and not force:
        print(file_name, 'already exists.')
        return
    np.save(file_name, array)


def scaleData(array, depth):
    """
    Scales values of elements in an array that range between [0, depth] into an array with elements between [-0.5, .05]

    Inputs:
    - array: a numpy array with elements between [0, depth]
    - depth: depth of values (e.g. maximum value that any element is allowed to have)

    Output:
    - An array with the same shape of input array with elements between [-0.5, 0.5]
    """
    return array / depth - .5


def showMultipleArraysHorizontally(array, labels=None, max_per_row=10):
    """
    Takes an array with the shape [number of images, width, height] and shows the images in rows.

    Inputs:
    - array: a 3 dimensional array with the shape: [number of images, width, height]
    - labels: a 1 dimensional array with the length of number of images in which each element is the label of corresponding image in input `array`.
    - max_per_row: maximum number of images in each row before going to the next row.
    """
    from matplotlib.pyplot import figure, imshow, axis
    # from matplotlib.image import imread
    # from random import sample
    # from matplotlib.pyplot import matshow
    import matplotlib.pyplot as plt
    fig = figure()
    number_of_images = array.shape[0]
    rows = np.ceil(number_of_images / max_per_row)
    columns = min(number_of_images, max_per_row)
    for i in range(number_of_images):
        ax = fig.add_subplot(rows, columns, i + 1)
        if labels is not None:
            ax.set_title(labels[i])
        plt.imshow(array[i])
        axis('off')
    plt.show()


# def rowNumbers(data, labels, number_of_digits):
#     """
#     Returns an image in form of array that contains multiple digits from MNIST dataset.
#     Inputs:
#     - number_of_digits: number of digits in the output image array.
#     - data: TODO
#     - labels: TODO

#     Output:
#     - A 2 dimensional array in which each element contains the value associated with a pixel.
#     """

#     # get `number_of_digits` sample images and their corresponding labels
#     random_index = np.random.choice(len(labels), number_of_digits)
#     random_data = data[random_index]
#     random_labels = labels[random_index]

#     # concatenate arrays to make a bigger image
#     new_array = np.concatenate(random_data, axis=1)

#     return new_array, random_labels


# def multipleNumberRows(data, labels, number_of_digits):
#     if isinstance(number_of_digits, int):
#         return rowNumbers(data, labels, number_of_digits)

#     total_data_count = len(labels)
#     sample_data = list()
#     sample_labels = list()
#     for number in number_of_digits:
#         random_index = np.random.choice(total_data_count, number)
#         sample_data.append(np.concatenate(data[random_index], axis=1))
#         sample_labels.append(labels[random_index])

#     return sample_data, sample_labels


# def fixedSizeMultipleNumberRows(data, labels, number_of_digits, max_digits):
#     """
#     The `multipleNumberRows` performs faster.
#     """
#     edge = data.shape[1]
#     batch_size = len(number_of_digits)
#     batch = np.zeros((batch_size, edge, max_digits * edge))
#     batch_labels = np.ones((batch_size, max_digits)) * -1
#     total_data_count = len(labels)
#     for i in range(batch_size):
#         number = number_of_digits[i]
#         random_index = np.random.choice(total_data_count, number)
#         batch[i, :, :edge *
#               number] = np.concatenate(data[random_index], axis=1)
#         batch_labels[i, :number] = labels[random_index]
#     return batch, batch_labels


# def insertImageArray(front_array, back_array):
#     front_shape = front_array.shape
#     back_shape = back_array.shape
#     position_0 = np.random.randint(back_shape[0] - front_shape[0])
#     position_1 = np.random.randint(back_shape[1] - front_shape[1])
#     back_array[position_0:front_shape[0] + position_0,
#                position_1:front_shape[1] + position_1] = front_array
#     return back_array


###############################################################################
if __name__ == "__main__":
    files = ["train", "test", "extra"]

    # download mnist files
    for file in files:
        maybeDownload(file)

    data_samples = np.random.randint(1, 400, size=20)
    print(data_samples)
    train_files, train_labels = getLabels(
        'train/digitStruct.mat', data_samples)
    print(train_files)
    print(train_labels)
    data = getImage(train_files, 'train/', shape=(100, 50))
    print(data.shape)
    print(parseLabel(train_labels[0]))
    # showMultipleArraysHorizontally(data[:15], train_labels[:15], 3)



    
    # generate multi-digit numbers from digits in dataset
    # images, labels = fixedSizeMultipleNumberRows(
    #     test_data, test_labels, np.random.randint(1,4,size = 20), 4)
    # digit_length = np.random.randint(1, 4, size=20)
    # images, labels = multipleNumberRows(test_data, test_labels, digit_length)

    # image = insertImageArray(images[0], np.zeros((500, 900)))
    # showMultipleArraysHorizontally([image], [labels[0]], 1)
