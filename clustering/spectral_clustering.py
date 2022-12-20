import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def get_dictionary_mean(dictionary_key, radius = 0.6):
    raw_features = []
    mean = sum(dictionary_key)/len(dictionary_key) #mean of features
    raw_features.append(mean)
    #for each point we can have more than 1 feature stacked in dictionary, so make a mean

    feat_temp = dictionary_key
    feat_temp_dist=[np.linalg.norm(robust_feature - mean) for robust_feature in dictionary_key]
    #distance of each feature to the mean

    n_new_feat = max([int(radius*len(feat_temp)), 1])
    # we catch just the ones near
    feat_temp_filtr = sorted(zip(feat_temp, feat_temp_dist), key= lambda x: x[1])[0:n_new_feat]
    feat_temp_filtr = [x[0] for x in feat_temp_filtr]                    # just add the ones near

    robust_mean = sum(feat_temp_filtr)/len(feat_temp_filtr)

    #new mean (robust mean)
    dino = robust_mean
    return dino

def get_spectral_clustering(dictionary):
    print('[doing the spectral clustering...]')
    beta = 0.3
    mapa = {}
    ls = []

    for n,key in enumerate(dictionary): #probleme is that some points have more the 1 feature
        mapa[n] = key
        d = get_dictionary_mean(dictionary[key])

        ls.append(d)


    ls = np.array(ls, dtype=object)


    A = cosine_similarity(ls, ls) #problem shape = (100,2,384) = (100,768)
    A = np.where(A - beta > 0, A, 1e-6)

    A = torch.tensor(A, device='cpu', dtype=torch.float)
    D = torch.diag(torch.sum(A, dim=-1))
    W = torch.linalg.inv(D) @ A
    
    v_prev = torch.sum(A, dim=-1) / torch.sum(A) #sum of lines divided by sum of matrix
    delta_prev = np.inf #'infinity'

    epsilon = np.inf*torch.ones_like(v_prev)
    while torch.linalg.norm(epsilon, ord=np.inf) > 1e-6: #0.3
        prod = torch.mv(W, v_prev)
        v_next = prod / torch.linalg.norm(prod, ord=1)

        delta_next = v_next - v_prev
        epsilon = delta_next - delta_prev #first run doesnt make sense
        delta_prev = delta_next #but is fixed here
        v_prev = v_next

    return mapa, v_next
