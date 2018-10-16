import tensorflow as tf

from task_1.tf_utils import weight_variable, bias_variable


class CNN:

    def __init__(self, in_height, in_width, out_size, l2_regularization=True):

        self.in_height = in_height
        self.in_width = in_width
        self.out_size = out_size

        self.l2_reg_coefficient = 1e-6
        self.l2_regularization = l2_regularization

        self.x = None
        self.y = None
        self.phase = None
        self.keep_prob = None
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.mse = None
        self.prediction = None

    def build_model(self):

        self.x = tf.placeholder(shape=[None, self.in_height, self.in_width], dtype=tf.float32,
                                name='x_in')

        self.y = tf.placeholder(shape=[None, self.out_size],
                                dtype=tf.float32, name='y_out')

        logits, prediction, l2_loss = self.network(self.x)
        self.prediction = tf.identity(prediction, 'prediction')

        self.loss = self.est_loss(self.y, logits)

        self.optimizer = self.optimize(self.loss, l2_loss)
        self.mse = self.est_mse(self.y, logits)
        self.accuracy = self.est_accuracy(self.y, prediction)[0]

    def network(self, inp):

        inp = tf.reshape(inp, shape=[-1, self.in_height, self.in_width, 1])

        conv_1 = tf.layers.conv2d(inputs=inp, filters=32, kernel_size=3, padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=2, strides=2)

        conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=32, kernel_size=3, padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)

        max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2)

        fc_inp = tf.contrib.layers.flatten(max_pool_2)

        fc_1 = tf.contrib.layers.fully_connected(fc_inp, num_outputs=128)

        fc_2 = tf.contrib.layers.fully_connected(fc_1, num_outputs=self.out_size, activation_fn=None)

        out = tf.nn.softmax(fc_2)

        return fc_2, out, 0

    def est_loss(self, labels, logits):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    def est_mse(self, labels, logits):
        predictions = tf.nn.softmax(logits)
        return tf.losses.mean_squared_error(labels=labels, predictions=predictions)

    def est_accuracy(self, labels, logits):

        return tf.metrics.accuracy(tf.argmax(labels, axis=1), tf.argmax(logits, axis=1))

    def optimize(self, loss, l2_loss):
        return tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss + l2_loss*self.l2_reg_coefficient)
