""" Module to measure classification performance over time """

import time
from sklearn.metrics import accuracy_score

def classification_time(x,y_true,std_slr,model):
  print(f"Classifing {x.shape[0]} samples")
  init_cpu = time.process_time()
  init_r = time.perf_counter()
  data_slr = std_slr.transform(x)
  y_pred = model.predict(data_slr)
  end_cpu = time.process_time()
  end_r = time.perf_counter()

  print(f"The classification took {end_cpu - init_cpu:.6f}[s] CPU, {end_r - init_r:.6f}[s] real")

  print(f"Classification accuracy of the training samples: {accuracy_score(y_true,y_pred)}")