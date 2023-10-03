""" Module to measure feature extraction performance over time """

import numpy as np
import time
import os


class FeatureExtractionTime:
  def __init__(self, data_dir,num_classes) -> None:
    self.files_paths = self.prepareFiles(data_dir,num_classes)

  def feature_extraction_single(self, filename_num, num_iterations,
                                callback_feature_extraction, descriptor=None):
    filename = self.files_paths[filename_num]  # Fixed frame

    time_frames_cpu = np.zeros(num_iterations)  # Number of iterations
    time_frames_r = time_frames_cpu.copy()

    for i in range(len(time_frames_cpu)):
      init_cpu = time.process_time()
      init_r = time.perf_counter()
      callback_feature_extraction(filename,descriptor)
      end_cpu = time.process_time()
      end_r = time.perf_counter()
      time_frames_cpu[i] = end_cpu - init_cpu  # CPU time
      time_frames_r[i] = end_r - init_r  # real time

    for i in range(len(time_frames_cpu)):
      print(f"The iteration {i} took {time_frames_cpu[i]:.6f}[s] CPU, {time_frames_r[i]:.6f}[s] real")

    print(f"The max value was: {np.max(time_frames_cpu):.6f}[s] CPU, {np.max(time_frames_r):.6f}[s] real")
    print(f"The min value was: {np.min(time_frames_cpu):.6f}[s] CPU, {np.min(time_frames_r):.6f}[s] real")
    print(f"The average was:   {np.mean(time_frames_cpu):.6f}[s] CPU, {np.mean(time_frames_r):.6f}[s] real")

  def feature_extraction_n(self, callback_feature_extraction, descriptor=None):
    # Skipped frames due to empty features
    skippted_frames = 0
    init_cpu = time.process_time()
    init_r = time.perf_counter()
    for filename in self.files_paths:
      features = callback_feature_extraction(filename,descriptor)
      if len(features) == 0: skippted_frames += 1
    end_cpu = time.process_time()
    end_r = time.perf_counter()

    print(f"{len(self.files_paths)} frames processed")
    print(f"{skippted_frames} frames were skipped")
    print(f"The feature extraction took {end_cpu - init_cpu:.6f}[s] CPU, {end_r - init_r:.6f}[s] real")

  def prepareFiles(self,rootpath,num_classes):
    """ Generates file path lists """
    files_paths = []
    classpath = sorted(os.listdir(rootpath))
    for i in range(num_classes):
      filenames = sorted(os.listdir(os.path.join(rootpath, classpath[i])))
      for filename in filenames:
        files_paths.append(os.path.join(rootpath, classpath[i], filename))
    return files_paths
