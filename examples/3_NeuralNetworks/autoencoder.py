# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.001 # 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
with tf.name_scope('input'):
    X = tf.placeholder("float", [None, n_input])

def random_variable(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))
def constant_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
    
layer_weights_shape = {
    'encoder1': [n_input, n_hidden_1],
    'encoder2': [n_hidden_1, n_hidden_2],
    'decoder1': [n_hidden_2, n_hidden_1],
    'decoder2': [n_hidden_1, n_input],
}
layer_biases_shape = {
    'encoder1': [n_hidden_1],
    'encoder2': [n_hidden_2],
    'decoder1': [n_hidden_1],
    'decoder2': [n_input],
}

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

# Building the encoder
def build_layer(layer_name, input_layer, act=tf.nn.tanh):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = random_variable(layer_weights_shape[layer_name])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = constant_variable(layer_biases_shape[layer_name])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.add(tf.matmul(input_layer, weights), biases)
            tf.summary.histogram('pre_activations', preactivate)
        with tf.name_scope('activations'):
            activations = act(preactivate)
            tf.summary.histogram('activations', activations)
    return activations

def encoder(input_layer):
    # Encoder Hidden layer with sigmoid activation #1
    encoder1 = build_layer('encoder1', input_layer)
    encoder2 = build_layer('encoder2', encoder1)
    return encoder2


# Building the decoder
def decoder(input_layer):
    # Encoder Hidden layer with sigmoid activation #1
    decoder1 = build_layer('decoder1', input_layer)
    decoder2 = build_layer('decoder2', decoder1)
    return decoder2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
tf.summary.scalar('cost', cost)

#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

merged = tf.summary.merge_all()
base_dir = '/tmp/tensorflow/autoenc/'
log_dir = base_dir + 'logs/'
train_dir = log_dir + 'train/'
save_path = log_dir + 'autoenc'
checkpoint = log_dir + 'checkpoint'
if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
tf.gfile.MakeDirs(train_dir)

#test_writer = tf.summary.FileWriter(log_dir + '/test')

saver = tf.train.Saver()
restore = False

# Launch the graph
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    if tf.gfile.Exists(checkpoint) and restore:
        to_load = tf.train.latest_checkpoint(log_dir)
        saver.restore(sess, to_load)
    else:
        sess.run(init)
        
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            summary, opt, c = sess.run([merged, optimizer, cost], feed_dict={X: batch_xs})
            if i == total_batch - 1:
                train_writer.add_summary(summary, epoch)
                saver.save(sess, save_path, global_step=epoch)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
    train_writer.close()
    #test_writer.close()
    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
