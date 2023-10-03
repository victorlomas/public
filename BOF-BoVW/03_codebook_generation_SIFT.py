""" Test script to measure codebook generation performance over time with SIFT """

from joblib import load
import sys
sys.path.append("SharedFunctions")
from BoVW import getCodebook

dscs_train = load("Data/SIFT/seq-01_train.joblib")

# Number of feature points used to obtain the codebook
num_samples=150000
# Codebook size
k=1024
print("Codebook generation SIFT")
getCodebook(dscs_train,k,num_samples,13)