from task_2.classifiers.SVM import SVM
# from task_2.correlation_filter import CorrelationFilter
from task_2.correlation_filter2 import CorrelationFilter
from task_2.entropy_filter import EntropyFilter
from task_2.dataset import Dataset

'''
    Filters for feature selection.

    - Correlation filter correlation_filter.py
        Смотрим попарные корреляции признаков, если корреляция выше порога - выкидываем признак.
        Result:
        Accuracy: 0.65, selected ids num: 2587

    - Correlation filter 2 correlation_filter2.py
        Смотрим корреляцию между метками класса и признаками. Берем те, где корреляция больше.
        Result:
        Accuracy: 0.68, selected ids num: 10

    - Entropy filter entropy_filter.py
        Accuracy: 0.67, selected ids num: 50


'''


if __name__ == '__main__':

    ds = Dataset()
    ds.create_ds()

    filterEngine = CorrelationFilter(num_feats=10)

    selected_ids = filterEngine.fit_data(ds.train_set['data'], ds.train_set['labels'])

    '''
    filterEngine = EntropyFilter(n=88)
    selected_ids = filterEngine.fit_data(ds.train_set['data'], ds.train_set['labels'])
    '''
    # test performance
    data, labels = ds.get_subset_set(selected_ids, 'train')
    valid_data, valid_labels = ds.get_subset_set(selected_ids, 'valid')
    svmEngine = SVM()

    svmEngine.fit(data, labels)

    print('Accuracy: {}, selected ids num: {}'.format(svmEngine.check_accuracy(valid_data, valid_labels),
                                                      len(selected_ids)))
