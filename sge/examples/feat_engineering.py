from operator import itemgetter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sge.utilities.protected_math import _log_, _exp_, _sqrt_, _div_, _inv_
from sge.parameters import params
import sge.grammar as grammar

class FeatEng():
    def __init__(self, problem = "bio"):
        self.invalid_fitness = 9999999999
        self.temp_info = []
        self.info = []
        self.temp_train = []
        self.mse_train = []

        if problem == "ld50":
            self.file = "resources/ld50.csv"
            self.target_column = "Column627"
        elif problem == "bio":
            self.file = "resources/bio.csv"
            self.target_column = "Column243"
        elif problem == "ppb":
            self.file = "resources/ppb.csv"
            self.target_column = "Column627"
        
        self.data = pd.read_csv(self.file, sep=';', encoding = 'utf-8').astype(np.float64)
        self.X = self.data.drop(self.target_column, axis = 1)
        self.y = self.data[self.target_column]

        self.train_set, self.test_set, self.y_train, self.y_test = train_test_split(self.X,self.y,
                                                                                test_size=0.5)


    def get_error(self, individual, dataset, target):
        # function = eval(individual, globals(), {"x": dataset})
        pred_error = 0.0
        try:
            # print(individual)
            # function = eval("lambda x: %s" % individual)
            # predicted = np.apply_along_axis(function, 1, dataset)
            predicted = eval(individual, globals(), {"x": dataset})
            pred_error = np.sum(np.power(predicted - target, 2))
        except (OverflowError, ValueError, SyntaxError, TypeError) as e:
            return self.invalid_fitness
        return pred_error



    def evaluate(self, individual):
        error = 0.0
        test_error = 0.0
        if individual == None:
            return self.invalid_fitness,  {'generation': 0, "evals": 1, "test_error": self.invalid_fitness}

        error = self.get_error(individual, self.train_set, self.y_train)
        error = _sqrt_(error / len(self.train_set))

        if error == None:
            error = self.invalid_fitness

        test_error = 0
        test_error = self.get_error(individual, self.test_set, self.y_test)
        test_error = _sqrt_(test_error / float(len(self.test_set)))

        return error,  {'generation': 0, "evals": 1, "test_error": test_error}


if __name__ == '__main__':
    import sge
    eval_func = FeatEng()
    sge.evolutionary_algorithm(evaluation_function=eval_func)