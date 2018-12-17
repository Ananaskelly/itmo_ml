import numpy as np


from task_2.dataset import Dataset
from task_2.correlation_filter import CorrelationFilter
from task_2.SVM import SVM


if __name__ == '__main__':

    ds = Dataset()
    ds.create_ds()

    filterEngine = CorrelationFilter()

    selected_ids = filterEngine.fit_data(ds.train_set['data'])

    # test perfomance
    data, labels = ds.get_subset_set(selected_ids, 'train')
    valid_data, valid_lables = ds.get_subset_set(selected_ids, 'valid')
    svmEngine = SVM()

    svmEngine.fit(data, lables)

    print('Accuracy: {}, selected ids: {}'.format(svmEngine.check_accuracy(valid_data, valid_lables), selected_ids))