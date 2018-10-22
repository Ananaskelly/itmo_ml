import numpy as np


class GeneticAlg:

    def __init__(self, full_range, subset_size=500, population_size=50, mutable=True, iter=1000, crossover_cf=0.3,
                 mt_coeff1=0.05, mt_coeff2=0.5):

        self.full_range = full_range
        self.subset_size = subset_size
        self.population_size = population_size
        self.mutable = mutable
        self.crossover_cf = crossover_cf
        self.crossover_am = int(subset_size*crossover_cf)

        self.mt_coeff1 = mt_coeff1
        self.mt_coeff2 = mt_coeff2

        self.max_iter = iter
        self.current_iter = 0
        self.current_generation = None

    def gen_first_generation(self):

        arr = []
        for i in range(self.population_size):

            arr.append(np.random.randint(low=0, high=self.full_range, size=self.subset_size))

        self.current_generation = np.array(arr)

    def range_population(self, scores):

        return np.flip(np.argsort(scores), axis=0)

    def crossover(self, scores):

        range_idx = self.range_population(scores)

        next_generation = self.current_generation[range_idx]

        for i in range(0, self.population_size - 1, 2):

            set_diff1 = np.setdiff1d(next_generation[i], next_generation[i+1])
            set_diff2 = np.setdiff1d(next_generation[i+1], next_generation[i])

            if set_diff1.shape[0] < self.crossover_am:
                buffer1 = set_diff1
            else:
                buffer1 = np.random.choice(set_diff1, self.crossover_am)

            if set_diff2.shape[0] < self.crossover_am:
                buffer2 = set_diff2
            else:
                buffer2 = np.random.choice(set_diff2, self.crossover_am)

            sw_id1 = np.random.choice(self.subset_size, buffer1.shape[0])
            sw_id2 = np.random.choice(self.subset_size, buffer2.shape[0])

            next_generation[i, sw_id1] = buffer1
            next_generation[i+1, sw_id2] = buffer2

        self.current_generation = next_generation

    def mutation(self):

        mt_ex_num = int(self.population_size*self.mt_coeff1)
        mt_size = int(self.subset_size*self.mt_coeff2)
        mt_ex_ids = np.random.choice(self.population_size, mt_ex_num)

        for id_ in mt_ex_ids:

            set_diff = np.setdiff1d(np.arange(self.subset_size), self.current_generation[id_])
            mt_ids = np.random.choice(set_diff, size=mt_size)
            sw_ids = np.random.choice(self.subset_size, size=mt_size)
            self.current_generation[id_][sw_ids] = mt_ids
