import statistics
from copy import deepcopy
import numpy as np


class BayesClassifier():
    def __init__(self, attributNames:list, laplace = 50):
        self.laplaceNumber = laplace
        self.classNumbers = {}
        self.prioriProb = {}
        self.means = []
        self.stdevs = []
        self.attributNames = attributNames
        self.continousStats = {}
        self.classes = {}
        self.classValues = []

        self.discreetAttributes = {}
        self.continousAttributes = {} #both dicts are like attributeName: dataColumn

        self.discreetNumbers = {}
        self.discreetProb = {}

    def fit(self, X, y):
        self.classValues = y
        self.classes = set(self.classValues)
        #set priory Prob
        for myClass in self.classes:
            self.classNumbers[myClass] = 0
        for value in self.classValues:
            self.classNumbers[value] = self.classValues.count(value)

        for myClass in self.classes:
            self.discreetNumbers[myClass] = {}
            for attribute, column in self.discreetAttributes.items():
                self.discreetNumbers[myClass][attribute] = {}
                columnValues = {row[column] for row in X}
                for value in columnValues:
                    self.discreetNumbers[myClass][attribute][value] = 0


    def gaussian_func(self, x, mean, sd):
        return 1./(np.sqrt(2.*np.pi)*sd)*np.exp(-np.power((x - mean)/sd, 2.)/2)

    def get_mean_and_stdev(self, attri_data):
        return statistics.mean(attri_data), statistics.stdev(attri_data)

    def devideAttributes(self, trainData:np.ndarray):
        maxNumberForDiscreet = 4
        discreet, continous = {}, {}
        for i in range(0, len(self.attributNames)-1):
            uniqueValues = {row[i] for row in trainData}
            if len(uniqueValues) > maxNumberForDiscreet:
                continous[self.attributNames[i]] = i
            else:
                discreet[self.attributNames[i]] = i
        self.discreetAttributes, self.continousAttributes = discreet, continous


    def train(self, trainData:np.ndarray):
        self.devideAttributes(trainData)
        X = [row[:-1] for row in trainData]
        y = [row[-1] for row in trainData]
        self.fit(X, y)
        for row in trainData:
            for attribute, column in self.discreetAttributes.items():
                myClass = row[-1]
                value = row[column]
                self.discreetNumbers[myClass][attribute][value] +=1
        self.discreetProb = deepcopy(self.discreetNumbers)

        for myClass in self.classes:
            self.prioriProb[myClass] = self.classNumbers[myClass] / len(trainData)
            for attribute, column in self.discreetAttributes.items():
                columnValues = {row[column] for row in trainData}
                for value in columnValues:
                    self.discreetProb[myClass][attribute][value] = (self.discreetNumbers[myClass][attribute][value] + self.laplaceNumber) / (self.classNumbers[myClass] + len(columnValues) * self.laplaceNumber)

            self.continousStats[myClass] = {}
            for attribute, column in self.continousAttributes.items():
                values = [row[column] for row in trainData if row[-1] == myClass]
                med, stdev = statistics.mean(values), statistics.stdev(values)
                self.continousStats[myClass][attribute] = {}
                self.continousStats[myClass][attribute]["med"] = med
                self.continousStats[myClass][attribute]["stdev"] = stdev

    def predictRow(self, row):
        probabilities = []
        values = []
        for myClass in self.classes:
            prob = self.prioriProb[myClass]
            for attribute, column in self.discreetAttributes.items():
                value = row[column]
                prob *= self.discreetProb[myClass][attribute][value]
            for attribute, column in self.continousAttributes.items():
                value = row[column]
                med = self.continousStats[myClass][attribute]["med"]
                stdev = self.continousStats[myClass][attribute]["stdev"]
                prob *= self.gaussian_func(value, med, stdev)
            values.append(myClass)
            probabilities.append(prob)
        bestIdx = [i for i in range(0, len(probabilities)) if probabilities[i] == max(probabilities)][0]
        predicted = values[bestIdx]
        return predicted

    def predict(self, X):
        predictions = []
        for row in X:
            predicted = self.predictRow(row)
            predictions.append(predicted)
        return predictions