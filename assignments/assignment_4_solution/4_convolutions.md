
Deep Learning
=============

Assignment 4
------------

Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`, we trained fully connected networks to classify [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) characters.

The goal of this assignment is make the neural network convolutional.


```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
```


```python
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 28, 28) (200000,)
    Validation set (10000, 28, 28) (10000,)
    Test set (10000, 28, 28) (10000,)


Reformat into a TensorFlow-friendly shape:
- convolutions need the image data formatted as a cube (width by height by #channels)
- labels as float 1-hot encodings.


```python
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 28, 28, 1) (200000, 10)
    Validation set (10000, 28, 28, 1) (10000, 10)
    Test set (10000, 28, 28, 1) (10000, 10)



```python
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
```

Let's build a small network with two convolutional layers, followed by one fully connected layer. Convolutional networks are more expensive computationally, so we'll limit its depth and number of fully connected nodes.


```python
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
```


```python
num_steps = 101

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
      sleep(1)
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 3.084810
    Minibatch accuracy: 18.8%
    Validation accuracy: 9.9%
    Minibatch loss at step 50: 1.743861
    Minibatch accuracy: 43.8%
    Validation accuracy: 53.2%
    Minibatch loss at step 100: 1.079280
    Minibatch accuracy: 62.5%
    Validation accuracy: 73.9%
    Test accuracy: 81.3%


---
Problem 1
---------

The convolutional model above uses convolutions with stride 2 to reduce the dimensionality. Replace the strides by a max pooling operation (`nn.max_pool()`) of stride 2 and kernel size 2.

---


```python
def train(num_steps,batch_size):

    train_size = train_labels.shape[0]
    patch_size = 5
    depth = 16
    num_hidden = 64
    report_step = num_steps//10

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
        dataset = tf.placeholder(
            tf.float32,
            shape=(None, image_size, image_size, num_channels))
        labels = tf.placeholder(
            tf.float32, shape=(None, num_labels))

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # generateLogit.
        def generateLogit(data):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [
                                    1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(pooled + layer1_biases)
            conv = tf.nn.conv2d(hidden, layer2_weights, [
                                1, 1, 1, 1], padding='SAME')
            pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [
                                    1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(pooled + layer2_biases)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(
                hidden, [-1, shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = generateLogit(dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # Predictions for the training, validation, and test data.
        model_output = tf.nn.softmax(logits)
        corrects = tf.equal(tf.argmax(model_output, 1), tf.argmax(labels, 1))
        performance = tf.reduce_mean(tf.cast(corrects, "float")) * 100


    def batchedAccuracy(complete_dataset, complete_labels):
        length = len(complete_dataset)
        performance = 0
        for batch_start in range(0, length, batch_size):
            feed_dict = {dataset: complete_dataset[batch_start:(batch_start + batch_size)],
                         labels: complete_labels[batch_start:(batch_start + batch_size)]}
            performance += corrects.eval(feed_dict=feed_dict).sum()
        accuracy = performance / length
        return accuracy * 100

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            # batch_labels = train_labels[offset:(offset + batch_size), :]
            random_indices = np.random.randint(train_size, size=batch_size)
            feed_dict = {dataset: train_dataset[random_indices],
                         labels: train_labels[random_indices]}
            _, l, train_performance = session.run(
                [optimizer, loss, performance], feed_dict=feed_dict)
            if (step % report_step == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' %
                      train_performance)
                print('Validation accuracy: %.1f%%' %
                      batchedAccuracy(valid_dataset, valid_labels))
        print('Test accuracy: %.1f%%' % batchedAccuracy(test_dataset, test_labels))

train(1001,16)
```

    Initialized
    Minibatch loss at step 0: 5.622566
    Minibatch accuracy: 12.5%
    Validation accuracy: 10.0%
    Minibatch loss at step 100: 1.698813
    Minibatch accuracy: 43.8%
    Validation accuracy: 62.0%
    Minibatch loss at step 200: 1.172716
    Minibatch accuracy: 60.9%
    Validation accuracy: 76.2%
    Minibatch loss at step 300: 0.957688
    Minibatch accuracy: 64.1%
    Validation accuracy: 80.8%
    Minibatch loss at step 400: 0.945980
    Minibatch accuracy: 69.5%
    Validation accuracy: 81.6%
    Minibatch loss at step 500: 0.893510
    Minibatch accuracy: 69.5%
    Validation accuracy: 82.0%
    Minibatch loss at step 600: 0.798694
    Minibatch accuracy: 72.7%
    Validation accuracy: 82.9%
    Minibatch loss at step 700: 0.753435
    Minibatch accuracy: 75.0%
    Validation accuracy: 82.7%
    Minibatch loss at step 800: 0.659596
    Minibatch accuracy: 78.9%
    Validation accuracy: 83.8%
    Minibatch loss at step 900: 0.954274
    Minibatch accuracy: 68.8%
    Validation accuracy: 83.8%
    Minibatch loss at step 1000: 0.811021
    Minibatch accuracy: 79.7%
    Validation accuracy: 84.0%
    Test accuracy: 90.6%


---
Problem 2
---------

Try to get the best performance you can using a convolutional net. Look for example at the classic [LeNet5](http://yann.lecun.com/exdb/lenet/) architecture, adding Dropout, and/or adding learning rate decay.

---


```python
def train(num_steps,batch_size):

    train_size = train_labels.shape[0]
    patch_size = 5
    depth = 16
    num_hidden = 100
    num_hidden_2 = 100
    report_step = num_steps//10

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
        dataset = tf.placeholder(
            tf.float32,
            shape=(None, image_size, image_size, num_channels))
        labels = tf.placeholder(
            tf.float32, shape=(None, num_labels))

        # Variables.
        keep_prob = tf.placeholder(tf.float32)

        global_step = tf.Variable(0,trainable=False)
        learning_rate = tf.train.exponential_decay(.5,global_step,report_step,.95)
        
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.constant(.05,shape=[depth]))
        
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(.05, shape=[depth]))
        
        layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(.05, shape=[num_hidden]))
        
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_hidden_2], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(.05, shape=[num_hidden_2]))

        layer5_weights = tf.Variable(tf.truncated_normal(
            [num_hidden_2, num_labels], stddev=0.1))
        layer5_biases = tf.Variable(tf.constant(.05, shape=[num_labels]))

        # generateLogit.
        def generateLogit(data):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [
                                    1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(pooled + layer1_biases)
            droped = tf.nn.dropout(hidden,keep_prob)
            conv = tf.nn.conv2d(droped, layer2_weights, [
                                1, 1, 1, 1], padding='SAME')
            pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [
                                    1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(pooled + layer2_biases)
            droped = tf.nn.dropout(hidden,keep_prob)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(
                droped, [-1, shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            droped = tf.nn.dropout(hidden,keep_prob)
            hidden = tf.nn.relu(tf.matmul(hidden,layer4_weights)+layer4_biases)
            droped = tf.nn.dropout(hidden,keep_prob)
            return tf.matmul(droped, layer5_weights) + layer5_biases

        # Training computation.
        logits = generateLogit(dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

        # Predictions for the training, validation, and test data.
        model_output = tf.nn.softmax(logits)
        corrects = tf.equal(tf.argmax(model_output, 1), tf.argmax(labels, 1))
        performance = tf.reduce_mean(tf.cast(corrects, "float")) * 100


    def batchedAccuracy(complete_dataset, complete_labels):
        length = len(complete_dataset)
        performance = 0
        for batch_start in range(0, length, batch_size):
            feed_dict = {dataset: complete_dataset[batch_start:(batch_start + batch_size)],
                         labels: complete_labels[batch_start:(batch_start + batch_size)],
                         keep_prob:1.0}
            performance += corrects.eval(feed_dict=feed_dict).sum()
        accuracy = performance / length
        return accuracy * 100

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            # batch_labels = train_labels[offset:(offset + batch_size), :]
            random_indices = np.random.randint(train_size, size=batch_size)
            feed_dict = {dataset: train_dataset[random_indices],
                         labels: train_labels[random_indices],
                         keep_prob:.7}
            _, l, train_performance = session.run(
                [optimizer, loss, performance], feed_dict=feed_dict)
            if (step % report_step == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' %
                      train_performance)
                print('Validation accuracy: %.1f%%' %
                      batchedAccuracy(valid_dataset, valid_labels))
        print('Test accuracy: %.1f%%' % batchedAccuracy(test_dataset, test_labels))

train(1001,128)
```

    Initialized
    Minibatch loss at step 0: 2.341913
    Minibatch accuracy: 16.4%
    Validation accuracy: 14.6%
    Minibatch loss at step 100: 0.903560
    Minibatch accuracy: 67.2%
    Validation accuracy: 77.2%
    Minibatch loss at step 200: 0.544230
    Minibatch accuracy: 82.0%
    Validation accuracy: 83.5%
    Minibatch loss at step 300: 0.687237
    Minibatch accuracy: 78.1%
    Validation accuracy: 85.1%
    Minibatch loss at step 400: 0.457008
    Minibatch accuracy: 85.2%
    Validation accuracy: 86.2%
    Minibatch loss at step 500: 0.397343
    Minibatch accuracy: 86.7%
    Validation accuracy: 87.0%
    Minibatch loss at step 600: 0.500095
    Minibatch accuracy: 85.2%
    Validation accuracy: 87.3%
    Minibatch loss at step 700: 0.364056
    Minibatch accuracy: 87.5%
    Validation accuracy: 87.6%
    Minibatch loss at step 800: 0.547180
    Minibatch accuracy: 85.9%
    Validation accuracy: 87.9%
    Minibatch loss at step 900: 0.552768
    Minibatch accuracy: 86.7%
    Validation accuracy: 87.7%
    Minibatch loss at step 1000: 0.611865
    Minibatch accuracy: 82.0%
    Validation accuracy: 88.2%
    Test accuracy: 94.0%

