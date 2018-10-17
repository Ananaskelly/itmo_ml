import xgboost
import numpy as np
from sklearn.metrics import accuracy_score

from task_1.dataset import Dataset


ds = Dataset('../data/train.csv', '../data/test.csv')

x_train, y_train, x_valid, y_valid = ds.load_ds_with_valid_not_one_hot()
x_test = ds.load_test()

model = xgboost.XGBClassifier(learning_rate=0.01, n_estimators=300, max_depth=9)
model.fit(x_train, y_train)

y_valid_prd = model.predict(x_valid)

print(accuracy_score(y_valid, y_valid_prd))

y_prd = model.predict(x_test)

ds.test_to_csv('out_xgboost.csv', y_prd)
