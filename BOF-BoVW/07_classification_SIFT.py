""" Test script to measure model classification performance over time with SIFT
"""

import numpy as np
from joblib import load
import sys
sys.path.append("SharedFunctions")
from classification import classification_time

# Input data load
y_test = load("Data/SIFT/targets_seq01_25percent_test.joblib")
vw_test = load("Data/SIFT/vw_seq01_25percent_test.joblib")

# Model load
model = load("Data/models/seq01_75percent_train_1024w_C0.01_linear_tfidf.joblib")
std_slr = load("Data/models/standard_scaler_seq01_75percent_train_1024w_C0.01_linear_tfidf.joblib")

print("SVM model classification using SIFT")
classification_time(vw_test,y_test,std_slr,model)