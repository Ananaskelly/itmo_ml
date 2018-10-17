import tensorflow as tf

from task_1.tf_utils import weight_variable, bias_variable


class SimpleFC:

    def __init__(self, in_size, out_size):

        self.in_size = in_size
        self.out_size = out_size

        self.x = None
        self.y = None
        self.phase = None
        self.keep_prob = None
        self.loss = None
        self.optimizer = None
        self.mse_loss = None
        self.prediction = None

        self.prediction_class = None

    def build_model(self):

        self.x = tf.placeholder(shape=[None, self.in_size], dtype=tf.float32,
                                name='x_in')

        self.y = tf.placeholder(shape=[None, self.out_size],
                                dtype=tf.float32, name='y_out')

        prediction = self.network(self.x)
        self.prediction = tf.identity(tf.nn.softmax(prediction), 'prediction')
        self.prediction_class = tf.argmax(prediction, axis=1)

        self.loss = self.est_loss(self.y, prediction)

        self.optimizer = self.optimize(self.loss)
        self.mse_loss = self.est_mse(self.y, prediction)

    def network(self, inp):

        weights_1 = weight_variable([self.in_size, 256])
        bias_1 = bias_variable([256])

        fc_1 = tf.nn.relu(tf.matmul(inp, weights_1) + bias_1)

        weights_2 = weight_variable([256, 128])
        bias_2 = bias_variable([128])

        fc_2 = tf.nn.relu(tf.matmul(fc_1, weights_2) + bias_2)

        weights_3 = weight_variable([128, 7])

        logits = tf.matmul(fc_2, weights_3)

        return logits

    def est_loss(self, labels, logits):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    def est_mse(self, labels, logits):
        predictions = tf.nn.softmax(logits)
        return tf.reduce_mean(tf.metrics.mean_squared_error(labels=labels, predictions=predictions))

    def optimize(self, loss):
        return tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
