""" Test script to measure model classification performance over time with BOF
"""

import numpy as np
from joblib import load
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import time
# Input data load
y_test = load("Data/BOF/seq-01targets_25percent_test_20layers_minratio0.01_binaryimg300.joblib")
vw_test = load("Data/BOF/vw_seq-01_25percentTEST_20layers_minratio0.01_binaryimg300.joblib")

print("SVM model classification using BOF")

# Model load
std_slr = load("Data/models/7scenes/BOF/standard_scaler_7scenes_1024w_20layers_C2.8642857142857143_rbf_tfidf_minratio0.01_binaryimg300.joblib")
model = load("Data/models/7scenes/BOF/model_7scenes_1024w_20layers_C2.8642857142857143_rbf_tfidf_minratio0.01_binaryimg300.joblib")

init_cpu = time.process_time()
init_r = time.perf_counter()

test = vw_test
test = std_slr.transform(test)
test_pred = model.predict(test)

end_cpu = time.process_time()
end_r = time.perf_counter()

print(f"The classification took {end_cpu - init_cpu:.6f}[s] CPU, {end_r - init_r:.6f}[s] real")

print(f"Classification accuracy of the training samples: {100*model.score(test, y_test)}")
