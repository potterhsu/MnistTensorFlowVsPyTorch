import tensorflow as tf


class Model(object):

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def inference(x, keep_prob=0.5):
        with tf.name_scope('conv1'):
            weight = Model._weight_variable([5, 5, 1, 32])
            bias = Model._bias_variable([32])
            conv1 = tf.nn.relu(Model._conv2d(x, weight) + bias)

        with tf.name_scope('pool1'):
            pool1 = Model._max_pool_2x2(conv1)

        with tf.name_scope('conv2'):
            weight = Model._weight_variable([5, 5, 32, 64])
            bias = Model._bias_variable([64])
            conv2 = tf.nn.relu(Model._conv2d(pool1, weight) + bias)

        with tf.name_scope('pool2'):
            pool2 = Model._max_pool_2x2(conv2)

        with tf.name_scope('fc1'):
            batch_size = x.get_shape()[0].value
            reshape = tf.reshape(pool2, [batch_size, -1])
            weight = Model._weight_variable([reshape.get_shape()[1].value, 1024])
            bias = Model._bias_variable([1024])
            fc1 = tf.nn.relu(tf.matmul(reshape, weight) + bias)

        with tf.name_scope('dropout'):
            dropout = tf.nn.dropout(fc1, keep_prob)

        with tf.name_scope('fc2'):
            weight = Model._weight_variable([1024, 10])
            bias = Model._bias_variable([10])
            fc2 = tf.matmul(dropout, weight) + bias

        return fc2

    @staticmethod
    def loss(logits, labels):
        cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits))
        return cross_entropy
