""" Test script to measure model training performance over time with BOF 
  using defined parameters
"""

from sklearn.model_selection import train_test_split
from Model_Training import ModelTrainingTime
from joblib import dump, load
# Input data load
targets = load("Data/BOF/seq-01targets_75percent_train_20layers_minratio0.01_binaryimg300.joblib")
vw_train = load("Data/BOF/vw_seq-01_75percentTRAIN_20layers_minratio0.01_binaryimg300.joblib")

print("vw_train_len= "+str(vw_train.shape))

mtt = ModelTrainingTime(x_train=vw_train,y_train=targets)

print("SVM model training using defined parameters, BOF")
mtt.train_def_params(C=2.150714285714286,kernel='rbf')