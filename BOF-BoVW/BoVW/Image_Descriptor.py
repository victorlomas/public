""" Prepares files and extract keypoints and descriptors """

import time
import cv2 as cv
from tqdm import tqdm
import os


def prepareFiles(rootpath):
    """ Returns the filename, its corresponding label_id and label per image """
    filespaths = []
    label_ids = []
    labels = []
    classpath = sorted(os.listdir(rootpath))
    for i in range(len(classpath)):
        filenames = sorted(os.listdir(os.path.join(rootpath, classpath[i])))
        for filename in filenames:
            filespaths.append(os.path.join(rootpath, classpath[i], filename))
            label_ids.append(i)
            labels.append(classpath[i])
    return (filespaths, label_ids, labels)


def getKeypointsDescriptors(filenames,descriptor):
    """ Given a list of filenames, extract their keypoints and descriptors """
    dataset_kpts = []
    dataset_des = []
    print("Extracting Local Descriptors")
    init = time.time()
    for filename in tqdm(filenames):
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        kpts, des = descriptor.detectAndCompute(img, None)
        dataset_kpts.append(kpts)
        dataset_des.append(des)
    end = time.time()
    print("Done in "+str(end-init)+" secs.")
    return (dataset_kpts, dataset_des)
