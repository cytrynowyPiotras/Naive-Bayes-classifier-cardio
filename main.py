import pandas as pd
import numpy as np
from BayesClassifier import BayesClassifier


def readCSV() -> tuple:
    """returns data array without first column and attribute names"""
    dataset = pd.read_csv('cardio_train.csv', sep=';', dtype=float)
    data = dataset.iloc[:, 1:].values
    attributeNames = list(dataset.columns.values)[1:]
    return data, attributeNames


def remove_outliners(df:np.ndarray):
    uppers = []
    lowers = []
    properData = []
    for columnIdx in range(0, len(df[0])):
        values = [row[columnIdx] for row in df]
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower = q1 - 2 * iqr
        upper = q3 + 2 * iqr
        uppers.append(upper)
        lowers.append(lower)
    for row in df:
        if dataInSize(lowers, uppers, row):
            properData.append(row)
    return properData


def dataInSize(lowers, uppers, row):
    for idx in range(0, len(uppers)):
        value = row[idx]
        if value < lowers[idx] or value > uppers[idx]:
            return False
    return True


def splitData(data, trainingRatio: float=0.9) -> tuple:
    """returns tuple of splited data"""
    dataLen = len(data)
    idx = int((trainingRatio)*dataLen)
    return data[:idx], data[idx:]


def getSplitingIdxs(ratio: int, len: int):
    myList = []
    distance = int(len/ratio)
    firstIdx = 0
    secIdx = distance
    for i in range(0, ratio):
        points = firstIdx, secIdx
        myList.append(points)
        firstIdx += distance
        secIdx += distance
    return myList


def n_cross_validation(dataset: np.ndarray,attribusteNames,  parts = 5):
    indexes = getSplitingIdxs(parts, len(dataset))
    counter = 1
    results = []
    for points in indexes:
        firstIdx, secIdx = points
        validSet = dataset[firstIdx:secIdx]
        trainSet = dataset[:firstIdx] + dataset[secIdx+1:]
        classifier = BayesClassifier(attribusteNames)
        classifier.train(trainSet)

        X = [row[:-1] for row in validSet]
        y = [row[-1] for row in validSet]
        predictions = classifier.predict(X)
        correct=0
        for i in range(0, len(y)):
            if predictions[i] == y[i]:
                correct+=1
        result = correct/len(y)

        print(f"nCross: {counter}/{parts} parts. result: {result}")
        results.append(result)
        counter += 1
    print(f"nCross parts: {parts}, avg result: {sum(results)/len(results)}")

def main():
    splitRatio = 0.9

    data, atributeNames = readCSV()
    data = remove_outliners(data)
    n_cross_validation(data, atributeNames)

    classifier = BayesClassifier(atributeNames)
    trainData, validData = splitData(data, splitRatio)
    classifier.train(trainData)

    X = [row[:-1] for row in validData]
    y = [row[-1] for row in validData]

    predictions = classifier.predict(X)
    correct=0
    for i in range(0, len(y)):
        if predictions[i] == y[i]:
            correct+=1
    print(f"Standard split result: {correct/len(y)}")

if __name__ == "__main__":
    main()