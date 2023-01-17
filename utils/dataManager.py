import librosa
import numpy as np
import os
import keras.utils
import math

def processAndSaveData(useMFCC = False):
    for f in os.listdir('recordings'):
            datum = processWave('recordings/' + f, useMFCC)
            f_out = 'recordings_processed/' + f.split('.')[0]
            np.save(f_out, datum)

def getProcessedData():
    path = 'recordings_processed'
    data = []
    labels = []
    for f in os.listdir(path):
        data.append(np.load(path + '/' + f))
        labels.append(int(f.split('_')[0]))

    # Shuffle data and labels
    randomOrder = np.random.permutation(len(data))
    data = np.asarray(data)[randomOrder]
    labels = keras.utils.to_categorical(np.asarray(labels)[randomOrder])

    # Split data for training and testing
    dataSplit = 0.8 
    splitPoint = math.floor(dataSplit * len(data))
    trainData = data[:splitPoint]
    trainLabels = labels[:splitPoint]
    testData = data[splitPoint:]
    testLabels = labels[splitPoint:]
    return trainData, trainLabels, testData, testLabels

def getExternalTestData(useMFCC):
    # This gets the data from the tests/ folder
    # This is data that is NOT found in the dataset used for training/testing, but rather data from the user in order
    #   to validate that the classifier also works on real input
    
    testData = []
    labels = []
    for f in os.listdir('tests'):
            datum = processWave('tests/' + f, useMFCC)
            testData.append(datum)
            labels.append(int(f[4]))
    
    return np.asarray(testData), np.asarray(labels)

def processWave(path, useMFCC):
    signal, _ = librosa.load(path, sr=8000, mono = True, duration=1)
    signal = np.pad(signal,((0, 8000-len(signal)),))
    if useMFCC:
        return librosa.feature.mfcc(y=signal, sr=8000, n_mfcc=20)
    return librosa.amplitude_to_db(librosa.feature.melspectrogram(y=signal, sr=8000, n_mels=20))
    

            