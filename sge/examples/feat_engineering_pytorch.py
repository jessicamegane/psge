import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import cos, sin
import torch
from sge.utilities.pytorchtest import *



class RMSE():
    def __init__(self, train_set, test_set):
        self.criterion = torch.nn.MSELoss()
        
    def calculate(self, predicted, dataset):
        pred_error = torch.sqrt(self.criterion(predicted, dataset.target))
        return pred_error


class RRSE():

    def __init__(self, train_set, test_set):
        self.criterion = torch.nn.MSELoss(reduction='sum')
        train_output_mean = torch.mean(train_set.target)
        self.__RRSE_train_denominator = self.criterion(train_set.target, train_output_mean)
        if test_set:
            test_output_mean = torch.mean(test_set.target)
            self.__RRSE_test_denominator = self.criterion(test_set.target, test_output_mean)

    def calculate(self, predicted, dataset):
        pred_error = self.criterion(predicted, dataset.target)

        if dataset.type == "test":
            error = torch.sqrt(torch.div(pred_error, self.__RRSE_test_denominator))
        else:
            error = torch.sqrt(torch.div(pred_error, self.__RRSE_train_denominator))
        return error

class Dataset():
    def __init__(self, dataset, target, type):
        dataset = torch.tensor(dataset.values, dtype=torch.float32, device=cur_dev)
        target = torch.tensor(target.values, dtype=torch.float32, device=cur_dev)    
        self.values = dataset
        self.target = target
        self.type = type

class FeatEng():
    def __init__(self, problem = "ppb", has_test_set=True, invalid_fitness=9999999):
        self.__invalid_fitness = invalid_fitness
        self.has_test_set = has_test_set
        self.__train_set = []
        self.__test_set = None
        if problem == "ld50":
            self.file = "resources/ld50.csv"
        elif problem == "bio":
            self.file = "resources/bio.csv"
        elif problem == "ppb":
            self.file = "resources/ppb.csv"
        self.read_dataset()
        self.__loss_function = RMSE(self.__train_set, self.__test_set)


    def read_dataset(self):
        dataset = pd.read_csv(self.file, sep=';', encoding = 'utf-8', skiprows=1, header=None).astype(np.float64)
        train_set, test_set, y_train, y_test = train_test_split(dataset.iloc[:, :-1],dataset.iloc[:, -1:],
                                                                                test_size=0.5)
        self.__train_set = Dataset(train_set, y_train, "train")
        self.__test_set = Dataset(test_set, y_test, "test")

    def get_error(self, individual, dataset):
        pred_error = 0.0
        try:
            predicted = eval(individual, globals(), {"x": dataset.values})

            pred_error = self.__loss_function.calculate(predicted, dataset)
            if pred_error.isnan() or pred_error.isinf():
                pred_error = torch.tensor(self.__invalid_fitness, device=cur_dev)

        except (OverflowError, ValueError) as e:
            return torch.tensor(self.__invalid_fitness, device=cur_dev)
        
        return pred_error
    

    def evaluate(self, individual):
        error = 0.0
        test_error = 0.0
        if individual is None:
            return None

        error = self.get_error(individual, self.__train_set)

        if self.__test_set:
            test_error = self.get_error(individual, self.__test_set)
            return (float(error.cpu().data.numpy()), {'generation': 0, "evals": 1, "test_error": float(test_error.cpu().data.numpy())})

        return (float(error.cpu().data.numpy()), {'generation': 0, "evals": 1, "test_error": test_error})


if __name__ == '__main__':
    import sge
    eval_func = FeatEng()
    sge.evolutionary_algorithm(evaluation_function=eval_func)