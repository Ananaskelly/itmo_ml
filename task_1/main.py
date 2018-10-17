import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


from task_1.dataset import Dataset
from task_1.model import SimpleFC


num_epoch = 250
batch_size = 28
print_step = 1000

ds = Dataset('../data/train.csv', '../data/test.csv')

x_train, y_train, x_valid, y_valid = ds.load_ds_with_valid()
y_valid_classes = ds.one_hot_to_dense(y_valid)

x_test = ds.load_test()

num_ex, num_feats = x_train.shape
num_steps = num_ex // batch_size

sess = tf.Session()

simple_fc = SimpleFC(in_size=num_feats, out_size=7)
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

        loss, _, mse_loss, prediction = sess.run([simple_fc.loss, simple_fc.optimizer, simple_fc.mse_loss,
                                                  simple_fc.prediction], feed_dict=feed_dict)

        if step % print_step == 0:
            print('Epoch {}, step {}, loss {}, mse {}'.format(epoch, step, loss, mse_loss))
            # print(y_train[step], prediction[0])

    feed_dict = {
        simple_fc.x: x_valid,
        simple_fc.y: y_valid,
    }

    mse_loss, class_predictions = sess.run([simple_fc.mse_loss, simple_fc.prediction_class], feed_dict=feed_dict)

    print('Epoch {}, mse {}'.format(epoch, mse_loss))
    print(accuracy_score(y_valid_classes, class_predictions))

feed_dict = {
    simple_fc.x: x_test,
}

num_tst_ex, _ = x_test.shape

prediction = sess.run([simple_fc.prediction_class], feed_dict=feed_dict)
prediction = np.reshape(prediction, newshape=num_tst_ex)

ds.test_to_csv('out_fc.csv', prediction)


