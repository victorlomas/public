""" Bag of Visual Words implementation """

import time
import numpy as np
import scipy.cluster.vq as vq


def getCodebook(descriptors_list, k, num_samples=None, seed=None):
    """ Generate the codebook

    Args:
        descriptors_list (list): List of descriptors
        k (int): Size of the codebook
        num_samples (int): Used to sample the descriptors if the amount of them
            is too big.
        seed (int): Seed for kmeans algorithm in order to be deterministic
    """
    descriptors = np.vstack(descriptors_list)
    if num_samples:
        np.random.seed(seed)
        descriptors = descriptors[
            np.random.choice(descriptors.shape[0], num_samples, replace=False)
        ]
    else:
        num_samples = descriptors.shape[0]
    print("Computing kmeans on "+str(num_samples) +
            " samples with "+str(k)+" centroids")
    init_cpu = time.process_time()
    init_r = time.perf_counter()
    codebook = vq.kmeans(obs=descriptors, k_or_guess=k, iter=6, seed=seed)[0]
    end_cpu = time.process_time()
    end_r = time.perf_counter()
    print(f"The codebook generation took {end_cpu - init_cpu:.6f}[s] CPU, {end_r - init_r:.6f}[s] real")
    return codebook


def getBoVWRepresentation(descriptors, codebook):
    """ Given a codebook, return the BoVW representation """
    num_imgs = len(descriptors)
    print(f"Processing histogram generation over {num_imgs} samples")
    k = codebook.shape[0]
    visual_words = np.zeros((num_imgs, k), dtype=np.float32)
    init_cpu = time.process_time()
    init_r = time.perf_counter()
    for i in range(num_imgs):
        words = vq.vq(descriptors[i], codebook)[0]
        visual_words[i, :] = np.bincount(words, minlength=k)
    end_cpu = time.process_time()
    end_r = time.perf_counter()
    print(f"The histogram generation took {end_cpu - init_cpu:.6f}[s] CPU, {end_r - init_r:.6f}[s] real")
    return visual_words


def bows_to_tf_idf(bow_vectors,num_img_with_word,num_imgs):
    k=bow_vectors.shape[1]
    new_bow_vectors= np.zeros((num_imgs, k), dtype=np.float32)
    #+1 to avoid division by zero
    denominator_idf=num_img_with_word+1
    for i in range(num_imgs):
        ocurrance_of_word=bow_vectors[i] 
        total_num_words=np.sum(bow_vectors[i])
        
        new_bow_vectors[i]=(ocurrance_of_word/total_num_words)*np.log(num_imgs/denominator_idf)
    return new_bow_vectors 


def getBoVWRepresentation_and_weights(descriptors, codebook):
    """ Given a codebook, return the BoVW representation """
    num_imgs = len(descriptors)
    k = codebook.shape[0]
    visual_words = np.zeros((num_imgs, k), dtype=np.float32)
    num_img_with_word=np.zeros(k)
    print(num_imgs)
    for i in range(num_imgs):
        words = vq.vq(descriptors[i], codebook)[0]
        visual_words[i, :] = np.bincount(words, minlength=k)
        
        sum_aux=np.where(visual_words[i, :]==0,0,1)
        num_img_with_word+=sum_aux
    print(num_img_with_word)
    denominator=num_img_with_word+1
    weights_tf_idf=np.log(num_imgs/denominator)

    return visual_words, weights_tf_idf

def get_tf_idf(descriptors,codebook,weights_tf_idf):
    num_imgs = len(descriptors)
    k = codebook.shape[0]
    visual_words_tf_idf = np.zeros((num_imgs, k), dtype=np.float32)
    num_img_with_word=np.zeros(k)
    
    for i in range(num_imgs):
        words = vq.vq(descriptors[i], codebook)[0]
        visual_words_tf_idf[i, :] = np.bincount(words, minlength=k)
        ocurrance_of_word=np.bincount(words, minlength=k)
        total_num_words=np.sum(ocurrance_of_word)
        visual_words_tf_idf[i, :]=(ocurrance_of_word/total_num_words)*weights_tf_idf
    
    return visual_words_tf_idf

def distance_between_words(u,v):
    u_unitary=u/np.linalg.norm(u,ord=1)
    v_unitary=v/np.linalg.norm(v,ord=1)
    
    distance=1-np.linalg.norm(u_unitary - v_unitary,ord=1)/2
    
    return distance
    
    
def normalized_distance_between_words(u,v,w):
    return distance_between_words(u,v)/distance_between_words(w,v)    
