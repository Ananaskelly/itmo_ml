import numpy as np

from task_2.GeneticAlg import GeneticAlg
from task_2.classifiers.RF import RF as CLF
from task_2.dataset import Dataset


'''
    Генетический алгоритм для задачи feature selection

    logs:

    [ 0.81  0.81  0.81  0.8   0.8   0.79  0.79  0.78  0.78  0.77]
    [ 0.84  0.82  0.82  0.8   0.8   0.8   0.8   0.8   0.78  0.78]
    [ 0.84  0.83  0.82  0.82  0.82  0.81  0.81  0.81  0.81  0.8 ]
    [ 0.85  0.84  0.83  0.83  0.83  0.82  0.82  0.82  0.82  0.82]
    [ 0.86  0.86  0.85  0.84  0.84  0.84  0.83  0.82  0.82  0.82]
    [ 0.89  0.87  0.84  0.84  0.84  0.84  0.84  0.83  0.83  0.83]
    [ 0.89  0.87  0.86  0.86  0.84  0.83  0.83  0.83  0.83  0.83]
    [ 0.91  0.87  0.86  0.85  0.85  0.85  0.84  0.83  0.83  0.83]
    [ 0.9   0.89  0.87  0.85  0.84  0.84  0.84  0.83  0.83  0.83]
    [ 0.86  0.86  0.86  0.85  0.84  0.83  0.83  0.83  0.83  0.83]

'''


if __name__ == '__main__':

    ds = Dataset()
    ds.create_ds()

    algEngine = GeneticAlg(full_range=ds.feats_num)
    algEngine.gen_first_generation()

    num_it = 50

    for i in range(num_it):
        ids_subsets = algEngine.current_generation

        scores = []
        imp_cfs = []
        for ids_subset in ids_subsets:

            data, labels = ds.get_subset_set(ids_subset)
            clfEngine = CLF()
            clfEngine.fit(data, labels)

            valid_data, valid_labels = ds.get_subset_set(ids_subset, data_type='valid')
            score, imp_cf = clfEngine.check_accuracy(valid_data, valid_labels)

            scores.append(score)
            imp_cfs.append(imp_cf)

        print(np.flip(np.sort(scores), axis=0)[:10])

        algEngine.crossover(scores, imp_cfs)
        algEngine.mutation()
