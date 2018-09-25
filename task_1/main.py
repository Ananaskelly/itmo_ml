import tensorflow as tf
import numpy as np


from task_1.dataset import Dataset
from task_1.model import SimpleFC


num_epoch = 1000
batch_size = 28
print_step = 100

ds = Dataset('../data/train.csv', '../data/test.csv')

x_train, y_train = ds.load_ds(ds_type='train')
num_ex, _ = x_train.shape
num_steps = num_ex // batch_size

sess = tf.Session()

simple_fc = SimpleFC(in_size=27, out_size=7)
simple_fc.build_model()

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

for epoch in range(num_epoch):

    perm = np.arange(num_ex)
    np.random.shuffle(perm)
    x_train = x_train[perm]
    y_train = y_train[perm]

    for step in range(0, num_ex, batch_size):

        feed_dict = {
            simple_fc.x: x_train[step: step + batch_size],
            simple_fc.y: y_train[step: step + batch_size],
        }

        loss, _, acc, prediction = sess.run([simple_fc.loss, simple_fc.optimizer, simple_fc.accuracy,
                                             simple_fc.prediction], feed_dict=feed_dict)

        if step % print_step == 0:
            print('Epoch {}, step {}, loss {}, mse {}'.format(epoch, step, loss, acc))
            print(y_train[step], prediction[0])



