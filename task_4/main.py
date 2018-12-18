from task_4.parser import Parser
from task_4.bayesian_calc import BayesianCalc

if __name__ == '__main__':

    parser = Parser('../data/cancer.bif')
    nodes = parser.parse_file()

    bayesianEngine = BayesianCalc(parser)

    line = input()
    result_prob = bayesianEngine.process_task(line)

    print('Result probability: {}'.format(result_prob))
