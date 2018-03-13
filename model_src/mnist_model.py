import time
import sys
import tensorflow as tf
import numpy as np

NUM_LABELS = 10
NUM_CHANNELS = 1
IMAGE_SIZE = 28
SEED = None


class Model:
    default_parameters = {
        'pixel_depth': 255,
        'batch_size': 64,
        'epochs': 10,
        'eval_batch_size': 64,
        'eval_frequency': 100,
        'learning_rate': 0.01
    }

    parameters = {}
    tf_session = None

    def __init__(self, parameters):
        self.parameters = self.default_parameters.copy()
        for k in parameters.keys():
            self.parameters[k] = parameters[k]

        self.conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32. stddev=0.1,
                                seed=SEED,
                                dtype=tf.float32),
                                name = 'conv1_weights')
        self.conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32), name = 'conv1_biases')
        self.conv2_weights = tf.Variable(tf.truncated_normal(
            [5, 5, 32, 64], stddev=0.1,
            seed=SEED, dtype=tf.float32),
            name = 'conv2_weights')
        self.conv2_biases = tf.Variable(
            tf.constant(0.1, shape=[64], dtype=tf.float32), name = 'conv2_biases')
        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                                stddev=0.1,
                                seed=SEED,
                                dtype=tf.float32),
                                name = 'fc1_weights')
        self.fc1_biases = tf.Variable(
            tf.constant(0.1, shape=[512], dtype=tf.float32),
            name = 'fc1_biases')
        self.fc2_weights = tf.Variable(
            tf.truncated_normal([512, NUM_LABELS],
                                stddev=0.1,
                                seed=SEED,
                                dtype=tf.float32),
                                name = 'fc2_weights')
        self.fc2_biases = tf.Variable(tf.constant(
            0.1, shape=[NUM_LABELS], dtype=tf.float32), name = 'fc2_biases')

    def _session (self):
        if self.tf_session is None:
            self.tf_session = tf.Session ()
            tf.global_variables_initializer().run(session = self.tf_session)

        return self.tf_session

    def _saver(self):
        saver = tf.train.Saver({
            "conv1_weights": self.conv1_weights,
            "conv1_biases": self.conv1_biases,
            "conv2_weights": self.conv2_weights,
            "conv2_biases": self.conv2_biases,
            "fc1_weights": self.fc1_weights,
            "fc1_biases": self.fc1_biases,
            "fc2_weights": self.fc2_weights,
            "fc2_biases": self.fc2_biases
        })

        return saver

    def save(self, model_path):
        saver = self._saver()

        sess = self._session ()
        saver.save(sess, model_path)

    def restore(self, model_path):
        saver = self._saver()

        sess = self._session ()
        saver.restore(sess, model_path)

    def _model(self, data, training):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            self.conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            self.conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
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
        hidden = tf.nn.relu(
            tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if training:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

    def train(self, train_data_set, eval_data_set):
        batch_size = self.parameters['batch_size']
        eval_batch_size = self.parameters['eval_batch_size']
        eval_frequency = self.parameters['eval_frequency']

        train_data = train_data_set['features']
        train_labels = train_data_set['labels']
        test_data = eval_data_set['features']
        test_labels = eval_data_set['labels']

        # Generate a validation set.
        validation_size = int(train_data.shape[0] * 0.2)
        validation_data = train_data[:validation_size, ...]
        validation_labels = train_labels[:validation_size]
        train_data = train_data[validation_size:, ...]
        train_labels = train_labels[validation_size:]
        num_epochs = self.parameters['epochs']

        train_size = train_labels.shape[0]

        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        train_data_node = tf.placeholder(
            tf.float32,
            shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        train_labels_node = tf.placeholder(tf.int64, shape=(batch_size,))

        # Training computation: logits + cross-entropy loss.
        logits = self._model(train_data_node, True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=train_labels_node, logits=logits))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                        tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0, dtype=tf.float32)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            # Base learning rate.
            self.parameters['learning_rate'],
            batch * batch_size,  # Current index into the dataset.
            train_size,          # Decay step.
            0.95,                # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, 0.9).minimize(loss, global_step=batch)

        # Predictions for the current training minibatch.
        train_prediction = tf.nn.softmax(logits)

        eval_data = tf.placeholder(
            tf.float32,
            shape=(eval_batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

        # Predictions for the test and validation, which we'll compute less often.
        eval_prediction = tf.nn.softmax(self._model(eval_data, False))

        def error_rate(predictions, labels):
            """Return the error rate based on dense predictions and sparse labels."""
            return 100.0 - (
                100.0 * np.sum(np.argmax(predictions, 1) == labels) /
                predictions.shape[0])

        # Small utility function to evaluate a dataset by feeding batches of data to
        # {eval_data} and pulling the results from {eval_predictions}.
        # Saves memory and enables this to run on smaller GPUs.
        def eval_in_batches(data, sess):
            """Get all predictions for a dataset by running it in small batches."""
            size = data.shape[0]
            if size < eval_batch_size:
                raise ValueError(
                    "batch size for evals larger than dataset: %d" % size)
            predictions = np.ndarray(
                shape=(size, NUM_LABELS), dtype=np.float32)
            for begin in xrange(0, size, eval_batch_size):
                end = begin + eval_batch_size
                if end <= size:
                    predictions[begin:end, :] = sess.run(
                        eval_prediction,
                        feed_dict={eval_data: data[begin:end, ...]})
                else:
                    batch_predictions = sess.run(
                        eval_prediction,
                        feed_dict={eval_data: data[-eval_batch_size:, ...]})
                    predictions[begin:,
                                :] = batch_predictions[begin - size:, :]
            return predictions

        # Create a local session to run the training.
        start_time = time.time()
        sess = self._session ()
        # Run all the initializers to prepare the trainable parameters.
        print('Initialized!')

        # Loop through training steps.
        for step in xrange(int(num_epochs * train_size) // batch_size):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * batch_size) % (train_size - batch_size)
            batch_data = train_data[offset:(offset + batch_size), ...]
            batch_labels = train_labels[offset:(offset + batch_size)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                        train_labels_node: batch_labels}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            # print some extra information once reach the evaluation frequency
            if step % eval_frequency == 0:
                # fetch some extra nodes' data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                            feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * batch_size / train_size,
                       1000 * elapsed_time / eval_frequency))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' %
                      error_rate(predictions, batch_labels))
                print('Validation error: %.1f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()
        # Finally print the result!
        test_error = error_rate(eval_in_batches(
            test_data, sess), test_labels)
        print('Test error: %.1f%%' % test_error)

    def predict(self, data):
        data_size = data.shape[0]
        eval_data = tf.placeholder(
            tf.float32,
            shape=(data_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

        # Predictions for the test and validation, which we'll compute less often.
        eval_prediction = tf.nn.softmax(self._model(eval_data, False))

        sess = self._session ()
        return np.argmax(sess.run(eval_prediction, feed_dict={eval_data: data}), 1)
