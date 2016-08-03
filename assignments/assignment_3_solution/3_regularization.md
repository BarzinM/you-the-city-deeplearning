
Deep Learning
=============

Assignment 3
------------

Previously in `2_fullyconnected.ipynb`, you trained a logistic regression and a neural network model.

The goal of this assignment is to explore regularization techniques.


```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
```

First reload the data we generated in _notmist.ipynb_.


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


Reformat into a shape that's more adapted to the models we're going to train:
- data as a flat matrix,
- labels as float 1-hot encodings.


```python
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
```

    Training set (200000, 784) (200000, 10)
    Validation set (10000, 784) (10000, 10)
    Test set (10000, 784) (10000, 10)



```python
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
```

---
Problem 1
---------

Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. The right amount of regularization should improve your validation / test accuracy.

---


```python
batch_size = 128
loss_coef = .004

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.add(tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)),loss_coef*tf.nn.l2_loss(weights))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
```


```python
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
#       print("Minibatch loss at step %d: %f" % (step, l))
#       print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Validation accuracy: 23.9%
    Validation accuracy: 83.2%
    Validation accuracy: 87.4%
    Validation accuracy: 87.8%
    Validation accuracy: 87.4%
    Validation accuracy: 88.2%
    Validation accuracy: 88.0%
    Test accuracy: 93.7%



```python
batch_size = 128
number_of_hidden_nodes = 1024
loss_coef = .0001


def generateWeight(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)


def generateBias(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_1 = generateWeight(
        [image_size * image_size, number_of_hidden_nodes])
    biases_1 = generateBias([number_of_hidden_nodes])

    layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
    loss_1 = loss_coef * tf.nn.l2_loss(weights_1)

    weights_2 = generateWeight([number_of_hidden_nodes, num_labels])
    biases_2 = generateBias([num_labels])

    logits = tf.matmul(layer_1, weights_2) + biases_2
    layer_2 = tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
    loss_2 = loss_coef * tf.nn.l2_loss(weights_2)
    loss = tf.add(tf.reduce_mean(layer_2), tf.add(loss_1, loss_2))

    # Training computation.
    # logits = tf.matmul(tf_train_dataset, weights) + biases
    # loss = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    layer_1_valid = tf.nn.relu(
        tf.matmul(tf_valid_dataset, weights_1) + biases_1)
    valid_logits = tf.matmul(layer_1_valid, weights_2) + biases_2
    valid_prediction = tf.nn.softmax(valid_logits)
    layer_1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
    test_logits = tf.matmul(layer_1_test, weights_2) + biases_2
    test_prediction = tf.nn.softmax(test_logits)

```


```python
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized",loss_coef)
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 100 == 0):
#       print("Minibatch loss at step %d: %f" % (step, l))
#       print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#       print("Validation accuracy: %.1f%%" % accuracy(
#         valid_prediction.eval(), valid_labels))
      print("%2.1f:%2.1f" %(accuracy(predictions, batch_labels),accuracy(valid_prediction.eval(), valid_labels)),end=" | ")

  print("\nTest accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized 0.0001
    5.5:31.1 | 79.7:82.2 | 81.2:82.9 | 85.9:84.6 | 85.9:85.3 | 89.1:83.0 | 85.9:85.4 | 85.2:86.4 | 80.5:85.1 | 89.8:86.5 | 85.9:87.3 | 
    Test accuracy: 92.7%



```python
def trainNetwork(loss_coef,train_size=None):
    if train_size is None:
        train_size = train_labels.shape[0]
    batch_size = 128
    number_of_hidden_nodes = 1024



    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(
            tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights_1 = generateWeight(
                [image_size * image_size, number_of_hidden_nodes])
        biases_1 = generateBias([number_of_hidden_nodes])

        layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
        loss_1 = loss_coef * tf.nn.l2_loss(weights_1)

        weights_2 = generateWeight([number_of_hidden_nodes, num_labels])
        biases_2 = generateBias([num_labels])

        logits = tf.matmul(layer_1, weights_2) + biases_2
        layer_2 = tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        loss_2 = loss_coef * tf.nn.l2_loss(weights_2)
        loss = tf.add(tf.reduce_mean(layer_2), tf.add(loss_1, loss_2))

        # Training computation.
        # logits = tf.matmul(tf_train_dataset, weights) + biases
        # loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        layer_1_valid = tf.nn.relu(
            tf.matmul(tf_valid_dataset, weights_1) + biases_1)
        valid_logits = tf.matmul(layer_1_valid, weights_2) + biases_2
        valid_prediction = tf.nn.softmax(valid_logits)
        layer_1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
        test_logits = tf.matmul(layer_1_test, weights_2) + biases_2
        test_prediction = tf.nn.softmax(test_logits)

    num_steps = 3001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized",loss_coef)
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
#           offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            offset = (step * batch_size) % (train_size - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 300 == 0):
                print("%2.1f:%2.1f" %(accuracy(predictions, batch_labels),accuracy(valid_prediction.eval(), valid_labels)),end=" | ")

        print("\nTest accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

for loss_coef in [1e-1,1e-2,1e-3,1e-4,1e-5]:
    trainNetwork(loss_coef)

```

    Initialized 0.1
    14.8:27.7 | 79.7:78.2 | 74.2:77.7 | 72.7:76.5 | 80.5:75.3 | 81.2:77.7 | 70.3:77.0 | 74.2:78.1 | 78.1:78.5 | 82.0:75.9 | 75.0:72.5 | 
    Test accuracy: 78.9%
    Initialized 0.01
    12.5:36.4 | 85.2:83.0 | 81.2:81.8 | 85.2:83.5 | 83.6:83.1 | 83.6:83.4 | 79.7:83.7 | 79.7:83.1 | 89.8:83.8 | 87.5:83.6 | 81.2:82.2 | 
    Test accuracy: 89.0%
    Initialized 0.001
    10.9:29.0 | 85.2:85.1 | 83.6:85.4 | 89.8:86.6 | 86.7:87.1 | 88.3:87.6 | 85.9:87.7 | 84.4:87.2 | 89.1:87.9 | 90.6:88.1 | 84.4:87.6 | 
    Test accuracy: 93.3%
    Initialized 0.0001
    7.0:27.0 | 82.0:84.8 | 82.8:85.4 | 89.1:86.6 | 85.9:86.9 | 86.7:87.7 | 87.5:88.0 | 84.4:87.3 | 88.3:88.3 | 89.8:88.9 | 86.7:88.0 | 
    Test accuracy: 93.9%
    Initialized 1e-05
    10.2:28.4 | 81.2:84.8 | 83.6:85.7 | 89.8:86.8 | 85.2:87.1 | 86.7:87.7 | 83.6:86.5 | 85.2:87.2 | 88.3:88.1 | 89.1:89.1 | 87.5:88.3 | 
    Test accuracy: 93.9%


---
Problem 2
---------
Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?

---


```python
trainNetwork(0,128*3)
```

    Initialized 0
    7.8:33.1 | 100.0:74.5 | 100.0:74.8 | 100.0:74.9 | 100.0:75.0 | 100.0:75.1 | 100.0:75.1 | 100.0:75.2 | 100.0:75.2 | 100.0:75.2 | 100.0:75.2 | 
    Test accuracy: 82.3%


---
Problem 3
---------
Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides `nn.dropout()` for that, but you have to make sure it's only inserted during training.

What happens to our extreme overfitting case?

---


```python
from time import time
def trainNetwork(loss_coef,train_size=None):
    if train_size is None:
        train_size = train_labels.shape[0]
    batch_size = 128
    number_of_hidden_nodes = 1024



    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(None, image_size * image_size))
        tf_train_labels = tf.placeholder(
            tf.float32, shape=(None, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights_1 = generateWeight(
                [image_size * image_size, number_of_hidden_nodes])
        biases_1 = generateBias([number_of_hidden_nodes])

        layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
        keep_prob = tf.placeholder(tf.float32)
        layer_1_drop = tf.nn.dropout(layer_1,keep_prob)
        loss_1 = loss_coef * tf.nn.l2_loss(weights_1)

        weights_2 = generateWeight([number_of_hidden_nodes, num_labels])
        biases_2 = generateBias([num_labels])

        logits = tf.matmul(layer_1_drop, weights_2) + biases_2
        layer_2 = tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        loss_2 = loss_coef * tf.nn.l2_loss(weights_2)
        loss = tf.add(tf.reduce_mean(layer_2), tf.add(loss_1, loss_2))

        # Training computation.
        # logits = tf.matmul(tf_train_dataset, weights) + biases
        # loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        layer_1_valid = tf.nn.relu(
            tf.matmul(tf_valid_dataset, weights_1) + biases_1)
        valid_logits = tf.matmul(layer_1_valid, weights_2) + biases_2
        valid_prediction = tf.nn.softmax(valid_logits)
        layer_1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
        test_logits = tf.matmul(layer_1_test, weights_2) + biases_2
        test_prediction = tf.nn.softmax(test_logits)

    num_steps = 301

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized",loss_coef)
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
#           offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            offset = (step * batch_size) % (train_size - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,keep_prob:.5}
            _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 300 == 0):
                print("%2.1f:%2.1f" %(accuracy(predictions, batch_labels),accuracy(valid_prediction.eval(feed_dict={keep_prob:1.0}), valid_labels)),end=" | ")

        print("\nTest accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels))
start=time()
trainNetwork(1e-4)
print('it took',time()-start)

```

    Initialized 0.0001
    5.5:8.7 | 11.7:10.0 | 
    Test accuracy: 10.0%
    it took 8.389643669128418


---
Problem 4
---------

Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is [97.1%](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).

One avenue you can explore is to add multiple layers.

Another one is to use learning rate decay:

    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
 
 ---



```python
from time import time

def trainNetwork(regularization_coef, num_steps, report_count=None, train_size=None):
    if report_count is None:
        report_count = num_steps // 10
    if train_size is None:
        train_size = train_labels.shape[0]
    batch_size = 128

    neuron_count_input = image_size * image_size
    neuron_count_1 = 400
    neuron_count_2 = 300
    neuron_count_output = num_labels

    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        keep_prob = tf.placeholder(tf.float32)
        global_step = tf.Variable(0,trainable=False)
        learning_rate = tf.train.exponential_decay(.5,global_step,report_count,.8)
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(None, neuron_count_input))
        tf_train_labels = tf.placeholder(
            tf.float32, shape=(None, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # layer 1: from input to first hidden layer
        weights_1 = generateWeight(
            [neuron_count_input, neuron_count_1])
        biases_1 = generateBias([neuron_count_1])

        layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
        layer_1_drop = tf.nn.dropout(layer_1, keep_prob)
        loss_1 = regularization_coef * tf.nn.l2_loss(weights_1)

        # layer 2: from first hidden layer to second
        weights_2 = generateWeight([neuron_count_1, neuron_count_2])
        biases_2 = generateBias([neuron_count_2])

        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
        layer_2_drop = tf.nn.dropout(layer_2, keep_prob)
        loss_2 = regularization_coef * tf.nn.l2_loss(weights_2)

        # layer 3: from second hidden layer to softmax
        weights_3 = generateWeight([neuron_count_2, neuron_count_output])
        biases_3 = generateBias([neuron_count_output])
        logits = tf.matmul(layer_2_drop, weights_3) + biases_3
        output_error = tf.nn.softmax_cross_entropy_with_logits(
            logits, tf_train_labels)
        loss_3 = regularization_coef * tf.nn.l2_loss(weights_3)

        # Optimizer.
        model_output = tf.nn.softmax(logits)
        corrects = tf.equal(tf.argmax(model_output, 1),
                            tf.argmax(tf_train_labels, 1))
        performance = tf.reduce_mean(tf.cast(corrects, "float"))*100
        loss = tf.reduce_mean(output_error) + loss_1 + loss_2 + loss_3
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)

    with tf.Session(graph=graph) as session:
        validation_dict = {tf_train_dataset: valid_dataset,
                           tf_train_labels: valid_labels,
                           keep_prob: 1.0}
        test_dict = {tf_train_dataset: test_dataset,
                     tf_train_labels: test_labels,
                     keep_prob: 1.0}

        tf.initialize_all_variables().run()
        print("Initialized", regularization_coef)
        for step in range(num_steps):

            # get random indices to generate minibatch.
            random_indices = np.random.randint(train_size,size=batch_size)

            # Prepare a dictionary telling the session where to feed the minibatch.
            train_dict = {tf_train_dataset: train_dataset[random_indices],
                          tf_train_labels: train_labels[random_indices],
                          keep_prob: .5}

            # Start the optimization
            _,train_performance,loss_value = session.run([optimizer,performance,loss], feed_dict=train_dict)

            # Generate report.
            if (step % report_count == 0):
                print("Minibatch loss, step %d: %f" %
                      (step, loss_value),end=" | ")
                valid_performance = performance.eval(feed_dict=validation_dict)
                print("%2.1f:%2.1f" % (train_performance, valid_performance))

        test_accuracy = performance.eval(feed_dict=test_dict)
        print("\nTest accuracy: %.1f%%" % test_accuracy)

start=time()
trainNetwork(1e-4, 15001)
print("it took",time()-start)
```

    Initialized 0.0001
    Minibatch loss, step 0: 4.292775 | 13.3:24.1
    Minibatch loss, step 1500: 0.584897 | 89.1:87.3
    Minibatch loss, step 3000: 0.377825 | 91.4:88.9
    Minibatch loss, step 4500: 0.534142 | 88.3:89.5
    Minibatch loss, step 6000: 0.454554 | 89.8:89.9
    Minibatch loss, step 7500: 0.523218 | 86.7:90.3
    Minibatch loss, step 9000: 0.620878 | 85.2:90.5
    Minibatch loss, step 10500: 0.444461 | 90.6:90.9
    Minibatch loss, step 12000: 0.421130 | 88.3:91.0
    Minibatch loss, step 13500: 0.437023 | 88.3:91.0
    Minibatch loss, step 15000: 0.303907 | 94.5:91.2
    
    Test accuracy: 95.8%
    it took 236.37441778182983

