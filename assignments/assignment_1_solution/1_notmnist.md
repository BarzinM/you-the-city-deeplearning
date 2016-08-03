
Deep Learning
=============

Assignment 1
------------

The objective of this assignment is to learn about simple data curation practices, and familiarize you with some of the data we'll be reusing later.

This notebook uses the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset to be used with python experiments. This dataset is designed to look like the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, while looking a little more like real data: it's a harder task, and the data is a lot less 'clean' than MNIST.


```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matlotlib backend as plotting inline in IPython
%matplotlib inline
```

First, we'll download the dataset to our local machine. The data consists of characters rendered in a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The training set has about 500k and the testset 19000 labelled examples. Given these sizes, it should be possible to train models quickly on any machine.


```python
url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
```

    Found and verified notMNIST_large.tar.gz
    Found and verified notMNIST_small.tar.gz


Extract the dataset from the compressed .tar.gz file.
This should give you a set of directories, labelled A through J.


```python
num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  print(data_folders)
  return data_folders
  
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
```

    notMNIST_large already present - Skipping extraction of notMNIST_large.tar.gz.
    ['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D', 'notMNIST_large/E', 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H', 'notMNIST_large/I', 'notMNIST_large/J']
    notMNIST_small already present - Skipping extraction of notMNIST_small.tar.gz.
    ['notMNIST_small/A', 'notMNIST_small/B', 'notMNIST_small/C', 'notMNIST_small/D', 'notMNIST_small/E', 'notMNIST_small/F', 'notMNIST_small/G', 'notMNIST_small/H', 'notMNIST_small/I', 'notMNIST_small/J']


---
Problem 1
---------

Let's take a peek at some of the data to make sure it looks sensible. Each exemplar should be an image of a character A through J rendered in a different font. Display a sample of the images that we just downloaded. Hint: you can use the package IPython.display.

---


```python
from IPython.display import Image,display
import os
from matplotlib.pyplot import figure,imshow,axis
from matplotlib.image import imread
from random import sample

directory = './notMNIST_small/'
    
def showImagesHorizontally(list_of_files):
    fig = figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(list_of_files[i])
        imshow(image,cmap='Greys_r')
        axis('off')

def showSampleImages(list_of_files,count=10,random=True):
    if random:
        files = sample(list_of_files,count)
    else:
        files = list_of_files[0:count]
    showImagesHorizontally(files)

def getSubdirectories(directory,append=False,sort=True):
    if append:
        subdirectories = [directory+name+'/' for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]
    else:
        subdirectories = [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]
    
    if sort:
        subdirectories = sorted(subdirectories)
    
    return subdirectories

def getFiles(directory,append=False,sort=True):
    if append:
        subdirectories = [directory+name for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))]
    else:
        subdirectories = [name for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))]
    
    if sort:
        subdirectories = sorted(subdirectories)
    
    return subdirectories

subdirectories = getSubdirectories(directory,True)
for subdirectory in subdirectories:
    showSampleImages(getFiles(subdirectory,True))


```


![png](output_7_0.png)



![png](output_7_1.png)



![png](output_7_2.png)



![png](output_7_3.png)



![png](output_7_4.png)



![png](output_7_5.png)



![png](output_7_6.png)



![png](output_7_7.png)



![png](output_7_8.png)



![png](output_7_9.png)


Now let's load the data in a more manageable format. Since, depending on your computer setup you might not be able to fit it all in memory, we'll load each class into a separate dataset, store them on disk and curate them independently. Later we'll merge them into a single dataset of manageable size.

We'll convert the entire dataset into a 3D array (image index, x, y) of floating point values, normalized to have approximately zero mean and standard deviation ~0.5 to make training easier down the road. 

A few images might not be readable, we'll just skip them.


```python
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)
```

    notMNIST_large/A.pickle already present - Skipping pickling.
    notMNIST_large/B.pickle already present - Skipping pickling.
    notMNIST_large/C.pickle already present - Skipping pickling.
    notMNIST_large/D.pickle already present - Skipping pickling.
    notMNIST_large/E.pickle already present - Skipping pickling.
    notMNIST_large/F.pickle already present - Skipping pickling.
    notMNIST_large/G.pickle already present - Skipping pickling.
    notMNIST_large/H.pickle already present - Skipping pickling.
    notMNIST_large/I.pickle already present - Skipping pickling.
    notMNIST_large/J.pickle already present - Skipping pickling.
    notMNIST_small/A.pickle already present - Skipping pickling.
    notMNIST_small/B.pickle already present - Skipping pickling.
    notMNIST_small/C.pickle already present - Skipping pickling.
    notMNIST_small/D.pickle already present - Skipping pickling.
    notMNIST_small/E.pickle already present - Skipping pickling.
    notMNIST_small/F.pickle already present - Skipping pickling.
    notMNIST_small/G.pickle already present - Skipping pickling.
    notMNIST_small/H.pickle already present - Skipping pickling.
    notMNIST_small/I.pickle already present - Skipping pickling.
    notMNIST_small/J.pickle already present - Skipping pickling.


---
Problem 2
---------

Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray. Hint: you can use matplotlib.pyplot.

---


```python
from matplotlib.pyplot import matshow
import matplotlib.pyplot as plt

def showMultipleArraysHorizontally(array,titles=None,max_per_row=10):
    fig = figure()
    number_of_images = len(array)
    for i in range(number_of_images):
        ax=fig.add_subplot(1+((number_of_images+1)//max_per_row),min(number_of_images,max_per_row),i+1)
        if titles is not None:
            ax.set_title(titles[i])
        plt.imshow(array[i], cmap=plt.get_cmap('gray'))
        axis('off')

for dataset in train_datasets:
    print('Dataset:',dataset)
    file = open(dataset,"rb")
    loaded_set=pickle.load(file)
    showMultipleArraysHorizontally(loaded_set[:20])
    print('Mean:',np.mean(loaded_set))
```

    Dataset: notMNIST_large/A.pickle
    Mean: -0.12825
    Dataset: notMNIST_large/B.pickle
    Mean: -0.00756304
    Dataset: notMNIST_large/C.pickle
    Mean: -0.142258
    Dataset: notMNIST_large/D.pickle
    Mean: -0.0573677
    Dataset: notMNIST_large/E.pickle
    Mean: -0.069899
    Dataset: notMNIST_large/F.pickle
    Mean: -0.125583
    Dataset: notMNIST_large/G.pickle
    Mean: -0.0945815
    Dataset: notMNIST_large/H.pickle
    Mean: -0.0685221
    Dataset: notMNIST_large/I.pickle
    Mean: 0.0307862
    Dataset: notMNIST_large/J.pickle
    Mean: -0.153358



![png](output_11_1.png)



![png](output_11_2.png)



![png](output_11_3.png)



![png](output_11_4.png)



![png](output_11_5.png)



![png](output_11_6.png)



![png](output_11_7.png)



![png](output_11_8.png)



![png](output_11_9.png)



![png](output_11_10.png)


---
Problem 3
---------
Another check: we expect the data to be balanced across classes. Verify that.

---

Answer: The mean is calculated in the previous code cell.

Merge and prune the training data as needed. Depending on your computer setup, you might not be able to fit it all in memory, and you can tune `train_size` as needed. The labels will be stored into a separate array of integers 0 through 9.

Also create a validation dataset for hyperparameter tuning.


```python
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
```

    Training: (200000, 28, 28) (200000,)
    Validation: (10000, 28, 28) (10000,)
    Testing: (10000, 28, 28) (10000,)


Next, we'll randomize the data. It's important to have the labels well shuffled for the training and test distributions to match.


```python
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
```

---
Problem 4
---------
Convince yourself that the data is still good after shuffling!

---


```python
def labelToChar(labels):
    key=['A','B','C','D','E','F','G','H','I','J']
    if isinstance(labels,(list,np.ndarray)):
        return [key[label] for label in labels]
    else:
        return key[labels]

def evaluateSets(dataset,labels):
    showMultipleArraysHorizontally(dataset,labelToChar(labels))
        
evaluation_count=10
showMultipleArraysHorizontally(train_dataset[:evaluation_count],labelToChar(train_labels[:evaluation_count]))
print('Mean:',np.mean(train_dataset))
showMultipleArraysHorizontally(test_dataset[:evaluation_count],labelToChar(test_labels[:evaluation_count]))
print('Mean:',np.mean(test_dataset))
showMultipleArraysHorizontally(valid_dataset[:evaluation_count],labelToChar(valid_labels[:evaluation_count]))
print('Mean:',np.mean(valid_dataset))
```

    Mean: -0.081835
    Mean: -0.0755267
    Mean: -0.0795241



![png](output_19_1.png)



![png](output_19_2.png)



![png](output_19_3.png)


Finally, let's save the data for later reuse:


```python
pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
```


```python
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
```

    Compressed pickle size: 690800503


---
Problem 5
---------

By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it.
Measure how much overlap there is between training, validation and test samples.

Optional questions:
- What about near duplicates between datasets? (images that are almost identical)
- Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
---


```python
import hashlib

temp_count = 10

def shrinkMatrices(ndarray, cell_size):
    # print('Shrinking array with shape:',ndarray.shape)
    dimensions = ndarray.shape
    if len(dimensions) == 2:
        output = np.reshape(ndarray, (dimensions[0] // cell_size,
                                      cell_size,
                                      dimensions[1] // cell_size,
                                      cell_size)).max(axis=(1, 3))
    else:
        output = np.reshape(ndarray, (dimensions[0], dimensions[1] // cell_size,
                                      cell_size,
                                      dimensions[2] // cell_size,
                                      cell_size)).max(axis=(2, 4))
    # print('Result array has the shape:',output.shape)
    return output

def hashArray(array):
    return [hashlib.sha1(x).digest() for x in array]


# Change the value of the variable below to remove only exact duplicates
# (False) or near duplicates (True). The variable below that (cell_size)
# controls how easy the algorithm will consider two elements similar. The
# higher the value of cell_size, the more elements identified as similar
remove_near_duplicates=True
cell_size = 4 # Or try 2, do not try 7 as it brings the bar for similarity too low

if remove_near_duplicates:
    # To remove near duplicates:
    train_dataset_hashed = hashArray(shrinkMatrices(train_dataset,cell_size))
    validation_dataset_hashed = hashArray(shrinkMatrices(valid_dataset,cell_size))
    test_dataset_hashed = hashArray(shrinkMatrices(test_dataset,cell_size))
    
else:
    # To remove exact duplicates:
    train_dataset_hashed = hashArray(train_dataset)
    validation_dataset_hashed = hashArray(valid_dataset)
    test_dataset_hashed = hashArray(test_dataset)


```


```python
## This part of code removes the elements in validation and test datasets
## that exist in other datasets. For example the test dataset elements that
## exist in either training dataset or validation dataset. Or, removing
## the elements in validation dataset that exist in training dataset.

test_unique = np.in1d(test_dataset_hashed, train_dataset_hashed,invert=True) &\
              np.in1d(test_dataset_hashed, validation_dataset_hashed,invert=True)

test_dataset_sanitized = test_dataset[test_unique]
test_labels_sanitized = test_labels[test_unique]
test_dataset_hashed_sanitized = [test_dataset_hashed[i] for i,x in enumerate(test_unique) if x]

print("%d samples removed from test" % (len(test_dataset)-len(test_dataset_sanitized)))

valid_unique = np.in1d(validation_dataset_hashed, train_dataset_hashed,invert=True)

valid_dataset_sanitized = valid_dataset[valid_unique]
valid_labels_sanitized = valid_labels[valid_unique]
validation_dataset_hashed_sanitized = [validation_dataset_hashed[i] for i,x in enumerate(valid_unique) if x]

print("%d samples removed from valid" % (len(valid_dataset)-len(valid_dataset_sanitized)))

```

    1603 samples removed from test
    1287 samples removed from valid



```python
def showCommons(hashed_array,original_array=[],labels=[],visualize=False):
    """This function is for visual analysis. It can show which elements 
    are indentified as duplicates within a given array. The first
    argument is needed but the second and third arguments are only used
    if the visualization is set to 'True'."""
    unique,unique_indices,unique_inverse,unique_counts=np.unique(hashed_array,True,True,True)
    print('In the array with',len(hashed_array),'elements, number of repeated data is:',len(hashed_array)-len(unique_indices))
    repeated=[]
    for reoccurance in np.where(unique_counts>1)[0]:
        repeated.append(np.where(unique_inverse==reoccurance))
    if visualize:
        for set in repeated:
            print('+ Indices with same data:',set[0],'-> only one is kept.')
            common_images=original_array[set]
            showMultipleArraysHorizontally(common_images,labelToChar(labels[set]))

def makeUnique(hashed_data,original_data,labels,return_hash=False):
    """Returns the unique elements and their corrisponding lables."""
    _,indices=np.unique(hashed_data,True)
    print('Removed %i elements.' %(len(hashed_data)-len(indices)))
    if return_hash:
        return (train_dataset[indices],
                train_labels[indices],
                [hashed_data[i] for i in indices])
    else:
        return train_dataset[indices], train_labels[indices]
                

# uncomment the line below to see a sample of duplicated images in test_dataset that will be removed
# showCommons(test_dataset_hashed[:2000],test_dataset[:2000],test_labels[:2000],True)

# testing to see if there is any duplicate left in training dataset after
# sanitization. The line below can be commented out without effecting the
# results.
# showCommons(train_dataset_hashed)

train_data_sanitized, train_labels_sanitized, train_dataset_hashed_sanitized = makeUnique(train_dataset_hashed,train_dataset,train_labels,True)
valid_data_sanitized, valid_labels_sanitized = makeUnique(validation_dataset_hashed_sanitized,valid_dataset_sanitized,valid_labels_sanitized)
test_data_sanitized, test_labels_sanitized = makeUnique(test_dataset_hashed_sanitized,test_dataset_sanitized,test_labels_sanitized)

# testing to see if there is any duplicate left in training dataset after
# sanitization. The line below can be commented out without effecting the
# results.
# showCommons(train_dataset_hashed_sanitized)
```

    Removed 17123 elements.
    Removed 20 elements.
    Removed 47 elements.


---
Problem 6
---------

Let's get an idea of what an off-the-shelf classifier can give you on this data. It's always good to check that there is something to learn, and that it's a problem that is not so trivial that a canned solution solves it.

Train a simple model on this data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression model from sklearn.linear_model.

Optional question: train an off-the-shelf model on all the data!

---


```python
from sklearn.linear_model import LogisticRegression

def flatIt(array):
    samples_in_array = array.shape[0]
    return np.reshape(array,(samples_in_array,-1))

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, warm_start=False)
data = flatIt(train_data_sanitized)
validation_data = flatIt(valid_data_sanitized)

sample_number_list = [50,100,1000,5000, len(data)]
for sample_number in sample_number_list:
    trained_model = model.fit(data[:sample_number],train_labels_sanitized[:sample_number])
    accuracy = trained_model.score(validation_data,valid_labels_sanitized)
    print('Training on %i samples resulted in accuray of %f' %(sample_number,accuracy))
```

    Training on 50 samples resulted in accuray of 0.567238
    Training on 100 samples resulted in accuray of 0.683538
    Training on 1000 samples resulted in accuray of 0.777982
    Training on 5000 samples resulted in accuray of 0.776602
    Training on 182877 samples resulted in accuray of 0.839756

