
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

  print("Train Labels")
  print(train_labels[:5])
```

    Training set (200000, 28, 28) (200000,)
    Validation set (10000, 28, 28) (10000,)
    Test set (10000, 28, 28) (10000,)
    Train Labels
    [4 9 6 2 7]


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

print("Train Labels")
print(train_labels[:5])
```

    Training set (200000, 784) (200000, 10)
    Validation set (10000, 784) (10000, 10)
    Test set (10000, 784) (10000, 10)
    Train Labels
    [[ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
     [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]
     [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]]



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

### Logistic model with Stochastic Gradient Descent


```python
batch_size = 128

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Regularization
  L2_reg = tf.nn.l2_loss(weights)
  beta = 1e-4
  loss += beta * L2_reg # Adding the value based on the regularization formula
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predict.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
```


```python
num_steps = 801

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 18.694584
    Minibatch accuracy: 5.5%
    Validation accuracy: 8.6%
    Minibatch loss at step 500: 1.914952
    Minibatch accuracy: 72.7%
    Validation accuracy: 75.4%
    Test accuracy: 84.0%


### Neural Network Model with Stochastic Gradient Descent


```python
batch_size = 128
num_nodes = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # 1st weights and biases
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_nodes]))
  biases1 = tf.Variable(tf.zeros([num_nodes]))
  
  # hidden layers generated by nn.relu()
  hidden = tf.nn.relu(tf.matmul(tf_train_dataset,weights1) + biases1)

  # 2nd weights and biases
  weights2 = tf.Variable(
    tf.truncated_normal([num_nodes, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
    
  # Training computation. (hidden layers and 2nd weigths and biases)
  logits = tf.matmul(hidden, weights2) + biases2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Regularization
  L2_reg = tf.nn.l2_loss(weights2)
  beta = 1e-4
  loss += beta * L2_reg # Adding the value based on the regularization formula
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1),weights2) + biases2)
  test_prediction = tf.nn.softmax(
    tf.matmul(
        tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1
                  ), weights2) + biases2)
```


```python
num_steps = 4001

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
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 341.360870
    Minibatch accuracy: 13.3%
    Validation accuracy: 28.1%
    Minibatch loss at step 500: 25.938923
    Minibatch accuracy: 79.7%
    Validation accuracy: 78.8%
    Minibatch loss at step 1000: 10.178591
    Minibatch accuracy: 76.6%
    Validation accuracy: 80.0%
    Minibatch loss at step 1500: 10.121580
    Minibatch accuracy: 71.9%
    Validation accuracy: 81.5%
    Minibatch loss at step 2000: 7.892292
    Minibatch accuracy: 84.4%
    Validation accuracy: 78.9%
    Minibatch loss at step 2500: 6.540093
    Minibatch accuracy: 89.1%
    Validation accuracy: 78.8%
    Minibatch loss at step 3000: 5.218542
    Minibatch accuracy: 80.5%
    Validation accuracy: 81.0%
    Minibatch loss at step 3500: 6.493916
    Minibatch accuracy: 78.9%
    Validation accuracy: 82.3%
    Minibatch loss at step 4000: 3.512716
    Minibatch accuracy: 81.2%
    Validation accuracy: 81.8%
    Test accuracy: 89.1%


---
Problem 2
---------
Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?

---


```python
batch_size = 128
num_nodes = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # 1st weights and biases
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_nodes]))
  biases1 = tf.Variable(tf.zeros([num_nodes]))
  
  # hidden layers generated by nn.relu()
  hidden = tf.nn.relu(tf.matmul(tf_train_dataset,weights1) + biases1)

  # 2nd weights and biases
  weights2 = tf.Variable(
    tf.truncated_normal([num_nodes, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
    
  # Training computation. (hidden layers and 2nd weigths and biases)
  logits = tf.matmul(hidden, weights2) + biases2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Regularization
  L2_reg = tf.nn.l2_loss(weights2)
  beta = 1e-4
  loss += beta * L2_reg # Adding the value based on the regularization formula
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1),weights2) + biases2)
  test_prediction = tf.nn.softmax(
    tf.matmul(
        tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1
                  ), weights2) + biases2)
```


```python
### Session part is identical except the offset part.

num_steps = 4001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (3 * batch_size)
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
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 255.580231
    Minibatch accuracy: 10.2%
    Validation accuracy: 30.8%
    Minibatch loss at step 500: 0.412531
    Minibatch accuracy: 100.0%
    Validation accuracy: 74.5%
    Minibatch loss at step 1000: 0.392411
    Minibatch accuracy: 100.0%
    Validation accuracy: 74.5%
    Minibatch loss at step 1500: 0.373272
    Minibatch accuracy: 100.0%
    Validation accuracy: 74.5%
    Minibatch loss at step 2000: 0.355067
    Minibatch accuracy: 100.0%
    Validation accuracy: 74.5%
    Minibatch loss at step 2500: 0.337750
    Minibatch accuracy: 100.0%
    Validation accuracy: 74.5%
    Minibatch loss at step 3000: 0.321277
    Minibatch accuracy: 100.0%
    Validation accuracy: 74.5%
    Minibatch loss at step 3500: 0.305608
    Minibatch accuracy: 100.0%
    Validation accuracy: 74.5%
    Minibatch loss at step 4000: 0.290703
    Minibatch accuracy: 100.0%
    Validation accuracy: 74.5%
    Test accuracy: 82.1%


The accuracy of minibatch is 100.0% suggesting that the the data have been overfitted. The result accuracy is slightly less than the previous result.

---
Problem 3
---------
Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides `nn.dropout()` for that, but you have to make sure it's only inserted during training.

What happens to our extreme overfitting case?

---


```python
""" 
The code is almost the same except the hidden layer part. 
Dropout has been added on the hidden layer
"""

batch_size = 128
num_nodes = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  # 1st weights and biases
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_nodes]))
  biases1 = tf.Variable(tf.zeros([num_nodes]))
  
  # hidden layers generated by nn.relu()
  hidden = tf.nn.relu(tf.matmul(tf_train_dataset,weights1) + biases1)
    
  # Dropout on the hidden layer
  hidden = tf.nn.dropout(hidden, 0.5)

  # 2nd weights and biases
  weights2 = tf.Variable(
    tf.truncated_normal([num_nodes, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
    
  # Training computation. (hidden layers and 2nd weigths and biases)
  logits = tf.matmul(hidden, weights2) + biases2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
  # Regularization
  L2_reg = tf.nn.l2_loss(weights2)
  beta = 1e-4
  loss += beta * L2_reg # Adding the value based on the regularization formula
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1),weights2) + biases2)
  test_prediction = tf.nn.softmax(
    tf.matmul(
        tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1
                  ), weights2) + biases2)
```


```python
### Session part is identical except the offset part.

num_steps = 4001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (3 * batch_size)
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
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

    Initialized
    Minibatch loss at step 0: 513.236511
    Minibatch accuracy: 12.5%
    Validation accuracy: 21.1%
    Minibatch loss at step 500: 0.444007
    Minibatch accuracy: 100.0%
    Validation accuracy: 76.7%
    Minibatch loss at step 1000: 0.425401
    Minibatch accuracy: 100.0%
    Validation accuracy: 77.0%
    Minibatch loss at step 1500: 0.406114
    Minibatch accuracy: 100.0%
    Validation accuracy: 77.1%
    Minibatch loss at step 2000: 0.388416
    Minibatch accuracy: 100.0%
    Validation accuracy: 77.6%
    Minibatch loss at step 2500: 0.370911
    Minibatch accuracy: 100.0%
    Validation accuracy: 76.8%
    Minibatch loss at step 3000: 0.354284
    Minibatch accuracy: 100.0%
    Validation accuracy: 77.2%
    Minibatch loss at step 3500: 0.337447
    Minibatch accuracy: 100.0%
    Validation accuracy: 77.1%
    Minibatch loss at step 4000: 0.321508
    Minibatch accuracy: 100.0%
    Validation accuracy: 77.2%
    Test accuracy: 84.6%


The result shows the better accuracy than the previous result where no dropout is added on the hidden layer.

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
### Try work on it later!!!!
```
