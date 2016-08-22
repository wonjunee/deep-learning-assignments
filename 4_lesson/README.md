
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
  # weights are used as filters
  # filter = [height, width, in_channels, out_channels]
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
    # tf.nn.conv2d(input, filter, strides, padding)
    # strides = [1, strid, strid, 1] <- 0th and 3rd are alwasys 1
    # padding = VALID (no padding), SAME (Zero paddings)
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
num_steps = 1001

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
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 4.348242
    Minibatch accuracy: 12.5%
    Validation accuracy: 10.0%
    Minibatch loss at step 50: 1.676310
    Minibatch accuracy: 56.2%
    Validation accuracy: 49.5%
    Minibatch loss at step 100: 1.644522
    Minibatch accuracy: 37.5%
    Validation accuracy: 63.0%
    Minibatch loss at step 150: 0.448405
    Minibatch accuracy: 87.5%
    Validation accuracy: 73.8%
    Minibatch loss at step 200: 0.631011
    Minibatch accuracy: 81.2%
    Validation accuracy: 77.6%
    Minibatch loss at step 250: 0.530011
    Minibatch accuracy: 87.5%
    Validation accuracy: 78.5%
    Minibatch loss at step 300: 0.699194
    Minibatch accuracy: 87.5%
    Validation accuracy: 78.7%
    Minibatch loss at step 350: 0.755387
    Minibatch accuracy: 68.8%
    Validation accuracy: 78.5%
    Minibatch loss at step 400: 0.606369
    Minibatch accuracy: 81.2%
    Validation accuracy: 78.9%
    Minibatch loss at step 450: 0.530153
    Minibatch accuracy: 81.2%
    Validation accuracy: 80.2%
    Minibatch loss at step 500: 0.565309
    Minibatch accuracy: 81.2%
    Validation accuracy: 80.2%
    Minibatch loss at step 550: 0.454898
    Minibatch accuracy: 81.2%
    Validation accuracy: 81.4%
    Minibatch loss at step 600: 0.291201
    Minibatch accuracy: 93.8%
    Validation accuracy: 81.7%
    Minibatch loss at step 650: 0.508770
    Minibatch accuracy: 87.5%
    Validation accuracy: 81.5%
    Minibatch loss at step 700: 0.876375
    Minibatch accuracy: 56.2%
    Validation accuracy: 81.5%
    Minibatch loss at step 750: 0.701382
    Minibatch accuracy: 75.0%
    Validation accuracy: 82.3%
    Minibatch loss at step 800: 0.647637
    Minibatch accuracy: 75.0%
    Validation accuracy: 82.7%
    Minibatch loss at step 850: 0.278678
    Minibatch accuracy: 87.5%
    Validation accuracy: 83.0%
    Minibatch loss at step 900: 0.264139
    Minibatch accuracy: 93.8%
    Validation accuracy: 82.9%
    Minibatch loss at step 950: 0.242353
    Minibatch accuracy: 93.8%
    Validation accuracy: 82.9%
    Minibatch loss at step 1000: 0.267299
    Minibatch accuracy: 87.5%
    Validation accuracy: 83.0%
    Test accuracy: 89.3%
    

---
Problem 1
---------

The convolutional model above uses convolutions with stride 2 to reduce the dimensionality. Replace the strides by a max pooling operation (`nn.max_pool()`) of stride 2 and kernel size 2.

---


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
  # weights are used as filters
  # filter = [height, width, in_channels, out_channels]
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
    # tf.nn.max_pool(data, kernel size, strides, padding)
    # ksize = [1,2,2,1] states that it is pooling the max over the 2x2 blocks.
    conv = tf.nn.max_pool(data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    conv = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
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
num_steps = 1001

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
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 3.925391
    Minibatch accuracy: 6.2%
    Validation accuracy: 10.0%
    Minibatch loss at step 50: 2.256890
    Minibatch accuracy: 18.8%
    Validation accuracy: 16.0%
    Minibatch loss at step 100: 2.094706
    Minibatch accuracy: 6.2%
    Validation accuracy: 21.0%
    Minibatch loss at step 150: 1.872543
    Minibatch accuracy: 37.5%
    Validation accuracy: 25.5%
    Minibatch loss at step 200: 2.125772
    Minibatch accuracy: 31.2%
    Validation accuracy: 16.9%
    Minibatch loss at step 250: 2.192461
    Minibatch accuracy: 18.8%
    Validation accuracy: 16.6%
    Minibatch loss at step 300: 2.081782
    Minibatch accuracy: 25.0%
    Validation accuracy: 29.4%
    Minibatch loss at step 350: 2.392615
    Minibatch accuracy: 37.5%
    Validation accuracy: 23.6%
    Minibatch loss at step 400: 1.769061
    Minibatch accuracy: 12.5%
    Validation accuracy: 40.4%
    Minibatch loss at step 450: 1.563624
    Minibatch accuracy: 50.0%
    Validation accuracy: 31.2%
    Minibatch loss at step 500: 1.798368
    Minibatch accuracy: 18.8%
    Validation accuracy: 45.0%
    Minibatch loss at step 550: 1.181898
    Minibatch accuracy: 75.0%
    Validation accuracy: 61.8%
    Minibatch loss at step 600: 1.070580
    Minibatch accuracy: 62.5%
    Validation accuracy: 56.6%
    Minibatch loss at step 650: 1.296888
    Minibatch accuracy: 68.8%
    Validation accuracy: 61.7%
    Minibatch loss at step 700: 1.807209
    Minibatch accuracy: 37.5%
    Validation accuracy: 64.2%
    Minibatch loss at step 750: 1.144567
    Minibatch accuracy: 62.5%
    Validation accuracy: 61.9%
    Minibatch loss at step 800: 1.186771
    Minibatch accuracy: 62.5%
    Validation accuracy: 64.9%
    Minibatch loss at step 850: 0.376730
    Minibatch accuracy: 81.2%
    Validation accuracy: 73.0%
    Minibatch loss at step 900: 0.824052
    Minibatch accuracy: 75.0%
    Validation accuracy: 73.6%
    Minibatch loss at step 950: 0.744368
    Minibatch accuracy: 75.0%
    Validation accuracy: 68.9%
    Minibatch loss at step 1000: 0.790194
    Minibatch accuracy: 81.2%
    Validation accuracy: 70.0%
    Test accuracy: 76.7%
    

The max pooling model is much faster than the convolutional model because max pooling model is only taking the max from the 2x2 blocks and no additional multiplication or addition of matrices are needed. However, the accuracy of the max pooling model is much lower than that of the convolutional model.

---
Problem 2
---------

Try to get the best performance you can using a convolutional net. Look for example at the classic [LeNet5](http://yann.lecun.com/exdb/lenet/) architecture, adding Dropout, and/or adding learning rate decay.

---

To improve the performance, I will use the convolution model instead of the max pool model. And I will add Dropout and learning rate decay to the model.

First, let's see the accuracy when I add Dropout only.


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
  # weights are used as filters
  # filter = [height, width, in_channels, out_channels]
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
    # tf.nn.conv2d(input, filter, strides, padding)
    # strides = [1, strid, strid, 1] <- 0th and 3rd are alwasys 1
    # padding = VALID (no padding), SAME (Zero paddings)
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    
    # Dropout on the hidden layer
    hidden = tf.nn.dropout(hidden, 0.5)
    
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
num_steps = 1001

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
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 4.504009
    Minibatch accuracy: 12.5%
    Validation accuracy: 10.5%
    Minibatch loss at step 50: 1.622739
    Minibatch accuracy: 50.0%
    Validation accuracy: 43.5%
    Minibatch loss at step 100: 1.353796
    Minibatch accuracy: 56.2%
    Validation accuracy: 64.2%
    Minibatch loss at step 150: 0.473635
    Minibatch accuracy: 81.2%
    Validation accuracy: 70.0%
    Minibatch loss at step 200: 0.777940
    Minibatch accuracy: 81.2%
    Validation accuracy: 73.6%
    Minibatch loss at step 250: 0.540854
    Minibatch accuracy: 93.8%
    Validation accuracy: 73.7%
    Minibatch loss at step 300: 0.785841
    Minibatch accuracy: 81.2%
    Validation accuracy: 76.4%
    Minibatch loss at step 350: 0.921049
    Minibatch accuracy: 68.8%
    Validation accuracy: 75.8%
    Minibatch loss at step 400: 0.988725
    Minibatch accuracy: 81.2%
    Validation accuracy: 75.3%
    Minibatch loss at step 450: 0.652972
    Minibatch accuracy: 75.0%
    Validation accuracy: 77.1%
    Minibatch loss at step 500: 0.755877
    Minibatch accuracy: 81.2%
    Validation accuracy: 77.0%
    Minibatch loss at step 550: 0.685099
    Minibatch accuracy: 81.2%
    Validation accuracy: 78.2%
    Minibatch loss at step 600: 0.399817
    Minibatch accuracy: 93.8%
    Validation accuracy: 79.7%
    Minibatch loss at step 650: 0.577900
    Minibatch accuracy: 81.2%
    Validation accuracy: 79.8%
    Minibatch loss at step 700: 1.027808
    Minibatch accuracy: 62.5%
    Validation accuracy: 79.7%
    Minibatch loss at step 750: 0.815430
    Minibatch accuracy: 75.0%
    Validation accuracy: 79.7%
    Minibatch loss at step 800: 0.800020
    Minibatch accuracy: 75.0%
    Validation accuracy: 80.3%
    Minibatch loss at step 850: 0.282923
    Minibatch accuracy: 87.5%
    Validation accuracy: 80.7%
    Minibatch loss at step 900: 0.281707
    Minibatch accuracy: 93.8%
    Validation accuracy: 81.2%
    Minibatch loss at step 950: 0.604451
    Minibatch accuracy: 75.0%
    Validation accuracy: 80.1%
    Minibatch loss at step 1000: 0.229818
    Minibatch accuracy: 93.8%
    Validation accuracy: 81.0%
    Test accuracy: 88.0%
    

The accuracy is slightly decreased when Dropout is added on the hidden layer.

Now let's add learning rate decay to the model.


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
  # weights are used as filters
  # filter = [height, width, in_channels, out_channels]
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
    # tf.nn.conv2d(input, filter, strides, padding)
    # strides = [1, strid, strid, 1] <- 0th and 3rd are alwasys 1
    # padding = VALID (no padding), SAME (Zero paddings)
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
    
  global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.1, global_step, 100000, .96, staircase=True)
    
  # Passing global_step to minimize() will increment it at each step.
  # so minimize() not only minimizes loss but also inclemnts global step by one after each update
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
```


```python
num_steps = 1001

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
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 3.427695
    Minibatch accuracy: 6.2%
    Validation accuracy: 14.1%
    Minibatch loss at step 50: 1.839630
    Minibatch accuracy: 43.8%
    Validation accuracy: 31.0%
    Minibatch loss at step 100: 1.731847
    Minibatch accuracy: 37.5%
    Validation accuracy: 63.3%
    Minibatch loss at step 150: 0.496733
    Minibatch accuracy: 81.2%
    Validation accuracy: 71.3%
    Minibatch loss at step 200: 0.591837
    Minibatch accuracy: 87.5%
    Validation accuracy: 77.1%
    Minibatch loss at step 250: 0.481184
    Minibatch accuracy: 93.8%
    Validation accuracy: 77.3%
    Minibatch loss at step 300: 0.706208
    Minibatch accuracy: 75.0%
    Validation accuracy: 78.1%
    Minibatch loss at step 350: 0.937920
    Minibatch accuracy: 68.8%
    Validation accuracy: 78.0%
    Minibatch loss at step 400: 0.611366
    Minibatch accuracy: 81.2%
    Validation accuracy: 78.6%
    Minibatch loss at step 450: 0.602472
    Minibatch accuracy: 81.2%
    Validation accuracy: 79.6%
    Minibatch loss at step 500: 0.717925
    Minibatch accuracy: 81.2%
    Validation accuracy: 78.6%
    Minibatch loss at step 550: 0.673993
    Minibatch accuracy: 81.2%
    Validation accuracy: 80.6%
    Minibatch loss at step 600: 0.352410
    Minibatch accuracy: 87.5%
    Validation accuracy: 80.2%
    Minibatch loss at step 650: 0.451789
    Minibatch accuracy: 81.2%
    Validation accuracy: 81.1%
    Minibatch loss at step 700: 0.882265
    Minibatch accuracy: 62.5%
    Validation accuracy: 81.5%
    Minibatch loss at step 750: 0.844494
    Minibatch accuracy: 75.0%
    Validation accuracy: 81.0%
    Minibatch loss at step 800: 0.630420
    Minibatch accuracy: 81.2%
    Validation accuracy: 81.6%
    Minibatch loss at step 850: 0.177732
    Minibatch accuracy: 100.0%
    Validation accuracy: 82.0%
    Minibatch loss at step 900: 0.273403
    Minibatch accuracy: 93.8%
    Validation accuracy: 82.3%
    Minibatch loss at step 950: 0.276992
    Minibatch accuracy: 93.8%
    Validation accuracy: 82.0%
    Minibatch loss at step 1000: 0.288935
    Minibatch accuracy: 93.8%
    Validation accuracy: 82.4%
    Test accuracy: 89.2%
    

The accuracy without the learning decay is 89.3% and now it is 89.2%. Let's try using Dropout and learning rate together.


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
  # weights are used as filters
  # filter = [height, width, in_channels, out_channels]
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
    # tf.nn.conv2d(input, filter, strides, padding)
    # strides = [1, strid, strid, 1] <- 0th and 3rd are alwasys 1
    # padding = VALID (no padding), SAME (Zero paddings)
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    
    # Dropout on the hidden layer
    hidden = tf.nn.dropout(hidden, 0.5)
    
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
    
  global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.1, global_step, 100000, .96, staircase=True)
    
  # Passing global_step to minimize() will increment it at each step.
  # so minimize() not only minimizes loss but also inclemnts global step by one after each update
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))
```


```python
num_steps = 1001

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
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 5.690289
    Minibatch accuracy: 6.2%
    Validation accuracy: 9.6%
    Minibatch loss at step 50: 2.135990
    Minibatch accuracy: 12.5%
    Validation accuracy: 25.5%
    Minibatch loss at step 100: 1.407874
    Minibatch accuracy: 50.0%
    Validation accuracy: 58.5%
    Minibatch loss at step 150: 0.726080
    Minibatch accuracy: 62.5%
    Validation accuracy: 65.3%
    Minibatch loss at step 200: 0.681207
    Minibatch accuracy: 87.5%
    Validation accuracy: 72.6%
    Minibatch loss at step 250: 0.832981
    Minibatch accuracy: 68.8%
    Validation accuracy: 72.7%
    Minibatch loss at step 300: 0.864824
    Minibatch accuracy: 68.8%
    Validation accuracy: 75.6%
    Minibatch loss at step 350: 0.975790
    Minibatch accuracy: 68.8%
    Validation accuracy: 74.7%
    Minibatch loss at step 400: 0.990926
    Minibatch accuracy: 62.5%
    Validation accuracy: 72.1%
    Minibatch loss at step 450: 0.502029
    Minibatch accuracy: 81.2%
    Validation accuracy: 76.3%
    Minibatch loss at step 500: 0.794286
    Minibatch accuracy: 81.2%
    Validation accuracy: 77.7%
    Minibatch loss at step 550: 0.686164
    Minibatch accuracy: 81.2%
    Validation accuracy: 78.3%
    Minibatch loss at step 600: 0.420769
    Minibatch accuracy: 87.5%
    Validation accuracy: 77.0%
    Minibatch loss at step 650: 0.516512
    Minibatch accuracy: 87.5%
    Validation accuracy: 79.3%
    Minibatch loss at step 700: 0.896898
    Minibatch accuracy: 68.8%
    Validation accuracy: 79.4%
    Minibatch loss at step 750: 0.678121
    Minibatch accuracy: 75.0%
    Validation accuracy: 79.1%
    Minibatch loss at step 800: 0.860501
    Minibatch accuracy: 68.8%
    Validation accuracy: 79.2%
    Minibatch loss at step 850: 0.307626
    Minibatch accuracy: 93.8%
    Validation accuracy: 79.9%
    Minibatch loss at step 900: 0.254725
    Minibatch accuracy: 100.0%
    Validation accuracy: 80.3%
    Minibatch loss at step 950: 0.466525
    Minibatch accuracy: 81.2%
    Validation accuracy: 79.2%
    Minibatch loss at step 1000: 0.269045
    Minibatch accuracy: 93.8%
    Validation accuracy: 80.4%
    Test accuracy: 87.3%
    

# Use 20001 steps for all models above!!!!
