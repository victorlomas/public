""" Test script to measure feature extraction performance over time with a single frame """

import cv2 as cv
import sys
sys.path.append("SharedFunctions")
from Feature_Extraction import FeatureExtractionTime


# Path of RGB images divided into classes
pcd_dir = "Data/Microsoft_7scenes/rgb"
num_classes = 1
filename_num = 568 # Fixed frame
num_iterations = 10 # Number of iterations

featureExtractionTime = FeatureExtractionTime(pcd_dir,num_classes)

def callback_sift(filename,descriptor):
  img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
  descriptor.detectAndCompute(img, None)

print("Single feature extraction SIFT")
descriptor = cv.SIFT_create()
featureExtractionTime.feature_extraction_single(filename_num, num_iterations,
                                callback_sift, descriptor)