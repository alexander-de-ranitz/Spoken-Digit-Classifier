from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense 
from utils import dataManager
from matplotlib import pyplot as plt
import numpy as np

def createNetwork(input_shape, num_classes):
    cnn = Sequential()

    cnn.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape, padding='same'))
    cnn.add(BatchNormalization())

    cnn.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same'))
    cnn.add(BatchNormalization())

    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))

    cnn.add(Dense(64, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(0.3))

    cnn.add(Dense(num_classes, activation='softmax'))
    cnn.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])

    return cnn

def trainAndEvaluateNetwork(batch_size=64, epochs=10, validation_split=0.1, useMFCC=True):
    trainData, trainLabels, testData, testLabels = dataManager.getProcessedData()

    inputShape = (trainData.shape[1], trainData.shape[2], 1)

    model = createNetwork(inputShape, 10)

    history = model.fit(trainData, trainLabels, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    print("---------------------------------")
    print("Evaluating model")
    testResults = model.evaluate(testData, testLabels)

    print("---------------------------------")
    print("Test results:")
    print(testResults)
    print("---------------------------------")

    # Get test data not from the original dataset- i.e. real world samples
    newTestData, newTestLabels = dataManager.getExternalTestData(useMFCC)

    predictions = model.predict(newTestData)
    correctPredictions = 0
    for i in range(len(predictions)):
        print("Prediction = ", np.argmax(predictions[i]), " | Actual = ", newTestLabels[i])
        if np.argmax(predictions[i]) == newTestLabels[i]:
            correctPredictions += 1
    print("The model correctly predicted ", str(correctPredictions), "/", str(len(newTestData)), " test samples!")
    print("")
    return history

def plotTrainingHistory(history):
    plt.rcParams.update({'font.size': 13})
    plt.plot(history.history['accuracy'], c='black', linestyle='--')
    plt.plot(history.history['val_accuracy'], c='black', linestyle='-')

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
