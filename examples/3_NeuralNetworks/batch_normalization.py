'''
Batch normalization with simple neural networks

Author: Aditya
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

print(mnist.count)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.6 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create model
def two_layer_new(x, weights, biases, dropout):

    fc1 = tf.add(tf.matmul(x, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    
    fc2 = tf.nn.dropout(fc2, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    return out

# Create model
def two_layer_batch_norm(x, weights, biases, dropout):
    
    epsilon = 1e-3
    z1_bn = tf.matmul(x,weights['wd1_bn'])
    
    batch_mean1, batch_var1 = tf.nn.moments(z1_bn,[0])
    
    z1_hat = (z1_bn - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
    
    bn1 = weights['scale1'] * z1_hat + weights['beta1']
    
    l_bn1 = tf.nn.relu(bn1)
    
    
    fc2 = tf.add(tf.matmul(l_bn1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    
    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    return out

def two_layer_batch_norm_2(x, weights, biases, dropout):
    
    epsilon = 1e-3
    z1_bn = tf.matmul(x,weights['wd1_bn'])
    
    batch_mean1, batch_var1 = tf.nn.moments(z1_bn,[0])
    
    z1_hat = (z1_bn - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
    
    bn1 = weights['scale1'] * z1_hat + weights['beta1']
    
    l_bn1 = tf.nn.relu(bn1)
    
    z2_bn = tf.matmul(l_bn1,weights['wd2_bn'])
    
    batch_mean2, batch_var2 = tf.nn.moments(z2_bn,[0])
    
    bn2 = tf.nn.batch_normalization(z2_bn,batch_mean2,batch_var2,weights['beta2'],weights['scale2'],epsilon)
    # Output, class prediction
    out = tf.add(tf.matmul(bn2, weights['out']), biases['out'])
    
    return out
# Store layers weight & bias
weights = {

    'wd1': tf.Variable(tf.random_normal([784, 256])),
    
    'wd2': tf.Variable(tf.random_normal([256, 128])),
    
    'out': tf.Variable(tf.random_normal([128, n_classes]))
}



biases = {
    'bd1': tf.Variable(tf.random_normal([256])),
    'bd2': tf.Variable(tf.random_normal([128])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

weights_BN = {
              'wd1_bn' : tf.Variable(tf.random_normal([784, 256])),
              'scale1' : tf.Variable(tf.ones([256])),
              'beta1' : tf.Variable(tf.zeros([256])),
              'wd2' : tf.Variable(tf.random_normal([256, 128])),
              'out' :tf.Variable(tf.random_normal([128, n_classes]))
              }

bias_BN = {
           'bd2' : tf.Variable(tf.random_normal([128])),
           'out': tf.Variable(tf.random_normal([n_classes]))
           }
weights_BN_2 = {
              'wd1_bn' : tf.Variable(tf.random_normal([784, 256])),
              'scale1' : tf.Variable(tf.ones([256])),
              'beta1' : tf.Variable(tf.zeros([256])),
              'wd2_bn' : tf.Variable(tf.random_normal([256, 128])),
              'scale2' : tf.Variable(tf.ones([128])),
              'beta2' : tf.Variable(tf.ones([128])),
              'out' :tf.Variable(tf.random_normal([128, n_classes]))
              }

bias_BN_2 = {
           'out': tf.Variable(tf.random_normal([n_classes]))
           }
# Construct model
#pred = two_layer_new(x, weights, biases, keep_prob)
pred = two_layer_batch_norm(x, weights_BN, bias_BN, keep_prob)
#pred = two_layer_batch_norm_2(x, weights_BN_2, bias_BN_2, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

training_epochs = 100

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    total_batch_train = int(mnist.train.num_examples/batch_size)
    total_batch_test = int(mnist.test.num_examples/batch_size)
    max_acc = -1.
    # Keep training until reach max iterations        
    for epoch in range(training_epochs):
        for i in range(total_batch_train):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            opt, c = sess.run([ optimizer, cost], 
                                   feed_dict={x: batch_xs,
                                              y: batch_ys,
                                              keep_prob: dropout})
        for i in range(total_batch_test):
            batch_xs, batch_ys = mnist.test.next_batch(batch_size)
                
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y: mnist.test.labels,
                                      keep_prob: 1.})
        if acc > max_acc:
            max_acc = acc
            print("New best testing Accuracy %.4f at epoch %d"%(acc, epoch))
        

        
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    

