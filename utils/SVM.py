from sklearn import svm
import numpy as np
from utils import dataManager

def trainAndEvaluateSVM(kernel="linear", decision_strategy = "ovo", useMFCC=True):
    trainData, trainLabels, testData, testLabels = dataManager.getProcessedData()
    trainData = np.asarray([features.flatten() for features in trainData])
    testData = np.asarray([features.flatten() for features in testData])

    trainLabels = np.asarray([str(np.argmax(oneHot)) for oneHot in trainLabels])
    testLabels = np.asarray([str(np.argmax(oneHot)) for oneHot in testLabels])

    res = svm.SVC(kernel=kernel, decision_function_shape=decision_strategy).fit(trainData, trainLabels)

    predictions = res.predict(testData)
    correctPredictions = 0
    for i in range(len(predictions)):
        if predictions[i] == testLabels[i]:
            correctPredictions += 1

    print("Predicted ", correctPredictions, " out of ", len(predictions), " correctly!")
    print("Accuracy = ", correctPredictions/len(predictions))

    newTestData, newTestLabels = dataManager.getExternalTestData(useMFCC)
    newTestData = np.asarray([features.flatten() for features in newTestData])
    newTestLabels = np.asarray([str(x) for x in newTestLabels])

    predictions = res.predict(newTestData)
    correctPredictions = 0
    for i in range(len(predictions)):
        print("Prediction = ",predictions[i], " | Actual = ", newTestLabels[i])
        if predictions[i] == newTestLabels[i]:
            correctPredictions += 1
    print("The model correctly predicted ", str(correctPredictions), "/", str(len(newTestData)), " novel test samples!")
    print("")
