""" Module to measure model training performance over time """


import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV

class ModelTrainingTime:
  def __init__(self,x_train,y_train) -> None:
    # Use standard scaler to standardize data
    self.stdSlr = StandardScaler()
    self.x_train = self.stdSlr.fit_transform(x_train)
    self.y_train = y_train
  
  def train_cv(self,folds,tune_kernel=True):
    if tune_kernel:
      tuned_parameters = [{'kernel': ["linear", "poly", "rbf", "sigmoid"]}]
    else:
      start=0.01
      end=10
      numparams=10
      tuned_parameters = [{'C':np.linspace(start,end,num=numparams)}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=folds,scoring='accuracy')
    init_cpu = time.process_time()
    init_r = time.perf_counter()
    clf.fit(self.x_train, self.y_train)
    end_cpu = time.process_time()
    end_r = time.perf_counter()

    print(f"The cross validation took {end_cpu - init_cpu:.6f}[s] CPU, {end_r - init_r:.6f}[s] real")

  def train_def_params(self,C,kernel):
    init_cpu = time.process_time()
    init_r = time.perf_counter()
    svm.SVC(C=C,kernel=kernel).fit(self.x_train,self.y_train)
    end_cpu = time.process_time()
    end_r = time.perf_counter()

    print(f"The training with defined parameters took {end_cpu - init_cpu:.6f}[s] CPU, {end_r - init_r:.6f}[s] real")