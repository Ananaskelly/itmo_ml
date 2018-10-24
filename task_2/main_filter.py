import numpy as np


from task_2.dataset import Dataset
from task_2.correlation_filter import CorrelationFilter
from task_2.SVM import SVM


if __name__ == '__main__':

    ds = Dataset()
    ds.create_ds()

    filterEngine = CorrelationFilter()

    selected_ids = filterEngine.fit_data(ds.train_set['data'])
    print(selected_ids)

    # test performance
    data, labels = ds.get_subset_set(selected_ids, 'train')
    valid_data, valid_labels = ds.get_subset_set(selected_ids, 'valid')
    svmEngine = SVM()

    svmEngine.fit(data, labels)

    print('Accuracy: {}, selected ids: {}'.format(svmEngine.check_accuracy(valid_data, valid_labels), selected_ids))
