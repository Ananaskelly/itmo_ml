import numpy as np

from task_2.GeneticAlg import GeneticAlg
from task_2.classifiers.RF import RF as CLF
from task_2.dataset import Dataset


if __name__ == '__main__':

    ds = Dataset()
    ds.create_ds()

    algEngine = GeneticAlg(full_range=ds.feats_num)
    algEngine.gen_first_generation()

    clfEngine = CLF()

    num_it = 1000

    for i in range(num_it):
        ids_subsets = algEngine.current_generation

        scores = []
        imp_cfs = []
        for ids_subset in ids_subsets:

            data, labels = ds.get_subset_set(ids_subset)
            clfEngine.fit(data, labels)

            valid_data, valid_labels = ds.get_subset_set(ids_subset, data_type='valid')
            score, imp_cf = clfEngine.check_accuracy(valid_data, valid_labels)

            scores.append(score)
            imp_cfs.append(imp_cf)

        print(np.flip(np.sort(scores), axis=0)[:10])

        algEngine.crossover(scores, imp_cfs)
        algEngine.mutation()
