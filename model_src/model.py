# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


# The variables below hold all the trainable weights. They are passed an
# initial value which will be assigned when we call:
# {tf.global_variables_initializer().run()}
conv1_weights = tf.Variable(
    tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                        stddev=0.1,
                        seed=SEED, dtype=tf.float32))
conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
conv2_weights = tf.Variable(tf.truncated_normal(
    [5, 5, 32, 64], stddev=0.1,
    seed=SEED, dtype=tf.float32))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
fc1_weights = tf.Variable(  # fully connected, depth 512.
    tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                        stddev=0.1,
                        seed=SEED,
                        dtype=tf.float32))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                              stddev=0.1,
                                              seed=SEED,
                                              dtype=tf.float32))
fc2_biases = tf.Variable(tf.constant(
    0.1, shape=[NUM_LABELS], dtype=tf.float32))

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.


def model(data, training):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if training:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases


def create_model(train_size, training=False):
    if training:
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        train_data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

        # Training computation: logits + cross-entropy loss.
        logits = model(train_data_node, training)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=train_labels_node, logits=logits))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0, dtype=tf.float32)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            0.01,                # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,          # Decay step.
            0.95,                # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               0.9).minimize(loss,
                                                             global_step=batch)

        # Predictions for the current training minibatch.
        train_prediction = tf.nn.softmax(logits)
        return (train_prediction, optimizer, loss, learning_rate, train_data_node, train_labels_node)
    else:
        eval_data = tf.placeholder(
            tf.float32,
            shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

        # Predictions for the test and validation, which we'll compute less often.
        eval_prediction = tf.nn.softmax(model(eval_data, training))

        return (eval_prediction, None, None, None, eval_data, None)