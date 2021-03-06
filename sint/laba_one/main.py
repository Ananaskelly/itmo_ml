import os
import logging
from sint.laba_one.dataset import Dataset

from task_2.classifiers.XGBOOST import XGBOOST
from task_2.classifiers.RF2 import RF

logger = logging.getLogger('laba_one')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

root_path = '../../data/'
file_name = 'subparts.xml'
ds = Dataset(os.path.join(root_path, file_name))
ds.create_ds(use_th_gr=True)

logger.info(msg='Dataset loaded!')

x, y = ds.train_set['data'], ds.train_set['labels']
x_test, y_test = ds.test_set['data'], ds.test_set['labels']
num_train_ex, feat_len = x.shape

logger.info(msg='Size of train set: {}, size of features in train set: {}'.format(num_train_ex, feat_len))

xgboostEngine = XGBOOST()
xgboostEngine.fit(x, y)

rfEngine = RF()
rfEngine.fit(x, y)

score = xgboostEngine.check_accuracy(x_test, y_test)
score_rf = rfEngine.check_accuracy(x_test, y_test)

logger.info(msg='Score on test set: {}'.format(score))
logger.info(msg='Score_rf on test set: {}'.format(score_rf))
