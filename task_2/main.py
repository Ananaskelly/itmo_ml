from task_2.dataset import Dataset
from task_2.GeneticAlg import GeneticAlg
from task_2.SVM import SVM
# from task_2.XGBOOST import XGBOOST


if __name__ == '__main__':

    ds = Dataset()
    ds.create_ds()

    algEngine = GeneticAlg(full_range=ds.feats_num)
    algEngine.gen_first_generation()

    clfEngine = SVM()

    num_it = 100

    for i in range(num_it):
        ids_subsets = algEngine.current_generation

        scores = []
        for ids_subset in ids_subsets:

            data, labels = ds.get_subset_set(ids_subset)
            clfEngine.fit(data, labels)

            valid_data, valid_labels = ds.get_subset_set(ids_subset, data_type='valid')
            scores.append(clfEngine.check_accuracy(valid_data, valid_labels))

        print(scores[:10])

        algEngine.crossover(scores)
        algEngine.mutation()
