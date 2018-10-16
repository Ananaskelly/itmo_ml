import tensorflow as tf
import numpy as np


from matveev.task_3.dataset import Dataset
from matveev.task_3.cnn import CNN


batch_size = 16
num_epoch = 100
print_step = 10


ds = Dataset()
x_train, y_train, x_test, y_test = ds.split_train_test_set()
num_ex, height, width = x_train.shape
num_steps = num_ex // batch_size

sess = tf.Session()

simple_cnn = CNN(in_height=height, in_width=width, out_size=40, l2_regularization=False)
simple_cnn.build_model()

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

for epoch in range(num_epoch):

    perm = np.arange(num_ex)
    np.random.shuffle(perm)
    x_train = x_train[perm]
    y_train = y_train[perm]

    for step in range(0, num_ex, batch_size):

        feed_dict = {
            simple_cnn.x: x_train[step: step + batch_size],
            simple_cnn.y: y_train[step: step + batch_size],
        }

        loss, _, acc, prediction = sess.run([simple_cnn.loss, simple_cnn.optimizer, simple_cnn.accuracy,
                                             simple_cnn.prediction], feed_dict=feed_dict)

        if step % print_step == 0:
            print('Epoch {}, step {}, loss {}, acc {}'.format(epoch, step, loss, acc))

feed_dict = {
    simple_cnn.x: x_test,
    simple_cnn.y: y_test
}

acc = sess.run([simple_cnn.accuracy], feed_dict=feed_dict)
print('TEST! acc {}'.format(acc))
