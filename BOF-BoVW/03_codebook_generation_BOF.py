""" Test script to measure codebook generation performance over time with BOF """

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import sys
sys.path.append("SharedFunctions")
from BoVW import getCodebook



filehandler = open("Data/BOF/seq-01_75percentTRAIN_3layers_minratio0.01_binaryimg300.bin", 'rb')
X_train = pickle.load(filehandler)
filehandler.close()


k = 1024 # Codebook size
print("Codebook generation BOF")
getCodebook(X_train,k,None,13)