""" Test script to measure model training performance over time with SIFT 
  using defined parameters
"""

import numpy as np
import sys
from joblib import dump, load
sys.path.append("SharedFunctions")
from Model_Training import ModelTrainingTime

# Input data load
y_train = load("Data/SIFT/targets_seq01_75percent_train.joblib")
vw_train = load("Data/SIFT/vw_seq01_75percent_train.joblib")


mtt = ModelTrainingTime(x_train=vw_train,y_train=y_train)

print("SVM model training using defined parameters, SIFT")
mtt.train_def_params(C=0.01,kernel='linear')