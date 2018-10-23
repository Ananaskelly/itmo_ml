import numpy as np


class SBS:

    def __init__(self, clf, dataset, stop_crit_it=3):

        self.stop_crit_it = stop_crit_it
        self.clf = clf()

        self.dataset = dataset

        self.current_ids = np.arange(self.dataset.feats_num)
        self.selected_ids = []

        self.last_f_it = -1
        self.current_it = 0

        self.last_score = 0
        self.current_score = 0

    def add_new_feature(self):

        scores = []

        for id_ in self.current_ids:
            curr_ids = np.array(self.selected_ids + [id_])
            data, labels = self.dataset.get_subset_set(curr_ids, 'train')
            self.clf.fit(data, labels)

            v_data, v_labels = self.dataset.get_subset_set(curr_ids, 'valid')
            scores.append(self.clf.check_accuracy(v_data, v_labels))

        scores = np.array(scores)

        self.last_score = self.current_score
        self.current_score = np.max(scores)
        winner_id = np.argmax(scores)

        winner_true_id = self.current_ids[winner_id]

        self.selected_ids.append(winner_true_id)
        self.current_ids = np.delete(self.current_ids, winner_id)

        self.current_it += 1

        return self.current_score

    def check_stopping_crit(self):

        if self.last_score > self.current_score:
            if self.last_f_it != -1:
                if self.current_it - self.last_f_it >= self.stop_crit_it:
                    return True
            else:
                self.last_f_it = self.current_it
        else:
            self.last_f_it = -1

        return False
