from task_4.node import Node
from task_4.factor import Factor
from task_4.parser import Parser
import copy
import numpy as np


def part_of(node, factor):
    for b in factor.get_fields():
        if node.get_name() == b.get_name():
            return True
    return False

if __name__ == '__main__':
    file_name = '../data/cancer.bif'
    p = Parser(file_name=file_name)
    bn = p.parse_file()
    factors = []
    for node in bn:
        if not node.is_root():
            temp_array = [node]
            temp_array.extend(node.get_parents())
            factors.append(Factor(node.get_dist(), temp_array))

    converged = False
    conver_num = 0

    while not converged:
        prev_conver_num = copy.deepcopy(conver_num)
        conver_num = 0
        for a in bn:
            for f in factors:
                if part_of(a, f):
                    message = a.send_marginal(f)
                    f.receive_belief(message, a)
        for f in factors:
            for a in bn:
                if part_of(a, f):
                    message = f.send_belief(a)
                    a.receive_marginal(message, f)
        for a in bn:
            a.update_marginal()
            conver_num += a.get_marginal()[list(a.get_marginal().keys())[0]]
        if np.abs(conver_num-prev_conver_num) < .00001:
            converged = True
    with open("results.txt", "w") as g:

        for a in bn:
            g.write(a.get_name() + " ")
            print(a.get_marginal())
            i = len(a.get_marginal().keys())-1
            while i >= 0:
                g.write(str(a.get_marginal()[list(a.get_marginal().keys())[i]]) + " ")
                i -= 1
            g.write("\n")
