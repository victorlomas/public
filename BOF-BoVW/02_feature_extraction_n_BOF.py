""" Test script to measure feature extraction performance over time with 1000 frames """

from extractBofs import extractBofs
import sys
sys.path.append("SharedFunctions")
from Feature_Extraction import FeatureExtractionTime


# Path of point clouds divided into classes
pcd_dir = "Data/Microsoft_7scenes/pcd"
num_classes = 1

featureExtractionTime = FeatureExtractionTime(pcd_dir,num_classes)

def callback_extractBofs(filename,_):
  #return extractBofs(filename,axis=2,method=2,layers=3)
  return extractBofs(pcd=filename,method=2,layers=20,axis=2,min_ratio=0.01,binary_size=300,plotBof=False,file=True)


print("Mutiple frames feature extraction BOF")
featureExtractionTime.feature_extraction_n(callback_extractBofs)
