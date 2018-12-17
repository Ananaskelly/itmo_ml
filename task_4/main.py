from task_4.parser import Parser
from itertools import product


def element_wise_equal(a, b):
    for i in range(0, len(a)):
        if a[i] != b[i]:
            return False
    return True


def get_dist_element(node, value, args_list):
    args = []
    for parent in node.parents:
        args.append(args_list[nodes.index(parent)])
    for key in node.dist:
        if type(key[0]) == tuple:
            conditions = key[1]
            values = key[0]
        else:
            conditions = []
            values = key
        if element_wise_equal(conditions, args):
            index = values.index(value)
            return node.dist[key][index]
    raise Exception('Unknown args: ' + str(args))


def get_marginal_dist(args):
    argument_ranges = []
    for node in nodes:
        if node.name in args:
            argument_ranges.append([args[node.name]])
        else:
            argument_ranges.append(node.states)
    total_prob = 0.0
    for values in product(*argument_ranges):
        total_product = 1.0
        for index in range(len(nodes)):
            value = values[index]
            element = get_dist_element(nodes[index], value, values)
            total_product *= element
        total_prob += total_product
    return total_prob


if __name__ == '__main__':

    parser = Parser('../data/asia.bif')
    nodes = parser.parse_file()

    line = input()
    parts = line.strip().split(',')
    args = {}
    for part in parts:
        if len(part) == 0:
            continue
        pp = part.strip().split('=')
        args[pp[0]] = pp[1]
    print(get_marginal_dist(args))
