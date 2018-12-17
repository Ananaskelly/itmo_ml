from itertools import product


class BayesianCalc:

    def __init__(self, parser):
        self.parser = parser
        self.nodes = self.parser.parse_file()

    def process_task(self, line):
        parts = line.strip().split(',')
        args = {}
        for part in parts:
            if len(part) == 0:
                continue
            pp = part.strip().split('=')
            args[pp[0]] = pp[1]

        return self.get_prob(args)

    def is_equal(self, a, b):
        for i in range(0, len(a)):
            if a[i] != b[i]:
                return False
        return True

    def get_element_prob(self, node, value, args_list):

        args = []
        for parent in node.parents:
            args.append(args_list[self.nodes.index(parent)])
        for key in node.dist:
            if type(key[0]) == tuple:
                conditions = key[1]
                values = key[0]
            else:
                conditions = []
                values = key
            if self.is_equal(conditions, args):
                index = values.index(value)
                return node.dist[key][index]
        raise Exception('Some problems!!')

    def get_prob(self, given_probs):
        argument_ranges = []
        for node in self.nodes:
            if node.name in given_probs:
                argument_ranges.append([given_probs[node.name]])
            else:
                argument_ranges.append(node.states)

        total_prob = 0.0
        for values in product(*argument_ranges):
            total_product = 1.0
            for index in range(len(self.nodes)):
                value = values[index]
                element = self.get_element_prob(self.nodes[index], value, values)
                total_product *= element
            total_prob += total_product
        return total_prob
