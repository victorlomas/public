""" Test script to measure performance of getting BOW tf-idf representation over time with BOF 
"""
import numpy as np
from joblib import dump, load
import time
from BoVW import *

init_cpu = time.process_time()
init_r = time.perf_counter()

bofs=load("Data/BOF/seq-01bofs_75percent_train_20layers_minratio0.01_binaryimg300.joblib")
codebook=np.load("Data/codebooks/7scenes/BOF/1024words_20layers_minratio0.01_binaryimg300.npy")
vw_train, weigths = getBoVWRepresentation_and_weights(bofs,codebook)
vw_tf_idf_train= get_tf_idf(bofs,codebook, weigths)

end_cpu = time.process_time()
end_r = time.perf_counter()

print(f"The obtention of the BOW-tfidf representation took {end_cpu - init_cpu:.6f}[s] CPU, {end_r - init_r:.6f}[s] real")