from task_2.classifiers.KNN import KNN
from task_2.classifiers.SVM import SVM

from task_2.SBS import SBS
from task_2.dataset import Dataset

if __name__ == '__main__':
    '''

        - KNN
            Current score: 0.99
            Selected features number: 12, ids: [8422, 2214, 2529, 6510, 2, 37, 6797, 51, 2622, 7157, 113]
        - SVM
            Current score: 0.92
            Selected features number: 15, ids: [964, 1760, 1175, 204, 242, 899, 1775, 8251, 4195, 236, 2, 22, 6313, 26]


    '''

    ds = Dataset()
    ds.create_ds()

    sbsEngine = SBS(clf=SVM, dataset=ds)

    sbsEngine.add_new_feature()

    while not sbsEngine.check_stopping_crit():
        current_score, selected_ids = sbsEngine.add_new_feature()
        print('Current score: {}, ids: {}'.format(current_score, selected_ids))

    print('Selected features number: {}, ids: {}'.format(sbsEngine.get_selected_features_num(),
                                                         sbsEngine.get_final_feats()))
