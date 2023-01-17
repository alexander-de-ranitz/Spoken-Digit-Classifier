from utils import dataManager
from utils import SVM
from utils import neuralNetwork

useMFCC = True #Whether to use MFCC or log-Mel spectrogram

# If the data is already stored, comment this line out to avoid recomputing it
dataManager.processAndSaveData(useMFCC=useMFCC)

history = neuralNetwork.trainAndEvaluateNetwork(useMFCC=useMFCC)
neuralNetwork.plotTrainingHistory(history)

testResults_svm_mfcc = SVM.trainAndEvaluateSVM(useMFCC=useMFCC)
