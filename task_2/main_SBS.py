import numpy as np


from task_2.dataset import Dataset
from task_2.SBS import SBS
from task_2.GeneticAlg import GeneticAlg
from task_2.SVM import SVM
from task_2.KNN import KNN
from task_2.RF import RF as CLF


if __name__ == '__main__':

    ds = Dataset()
    ds.create_ds()

    sbsEngine = SBS(clf=KNN, dataset=ds)

    sbsEngine.add_new_feature()

    while not sbsEngine.stop_crit_it():
        current_score = sbsEngine.add_new_feature()
        print('Current score: {}'.format(current_score))

    print('Selected features number: {}'.format(len(sbsEngine.selected_ids)))
