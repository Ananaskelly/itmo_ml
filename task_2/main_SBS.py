import numpy as np


from task_2.dataset import Dataset
from task_2.SBS import SBS
from task_2.GeneticAlg import GeneticAlg
from task_2.SVM import SVM
from task_2.KNN import KNN
from task_2.RF import RF
from task_2.XGBOOST import XGBOOST


if __name__ == '__main__':

    ds = Dataset()
    ds.create_ds()

    sbsEngine = SBS(clf=XGBOOST, dataset=ds)

    sbsEngine.add_new_feature()

    while not sbsEngine.check_stopping_crit():
        current_score, selected_ids = sbsEngine.add_new_feature()
        print('Current score: {}, ids: {}'.format(current_score, selected_ids))

    print('Selected features number: {}, ids: {}'.format(sbsEngine.get_selected_features_num(),
                                                         sbsEngine.get_final_feats()))
