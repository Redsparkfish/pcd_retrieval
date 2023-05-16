import numpy as np
import time
from sklearn.cluster import KMeans


def sim(clique_1, clique_2):
    return 1 - np.dot(clique_1, clique_2) / (np.linalg.norm(clique_1) * np.linalg.norm(clique_2))


def simplify(CSet: np.ndarray):
    size = CSet.shape[0]
    length = CSet.shape[1]
    bag = [CSet[0]]
    unique_indices = [0]
    for i in range(1, size):
        is_unique = True
        for j in unique_indices:
            if np.count_nonzero(CSet[i]) == np.count_nonzero(bag[j]):
                if np.all(CSet[i].nonzero()[0] == bag[j].nonzero()[0]):
                    CSet[i] = bag[j]
                    is_unique = False
                    break
            if sim(CSet[i], bag[j]) <= 0.134:
                CSet[i] = bag[j]
                is_unique = False
                break
        bag.append(CSet[i])
        if is_unique:
            unique_indices.append(i)
    return CSet, CSet[unique_indices]


tic = time.time()
CSet_collection = np.load('CSet_collection.npy', allow_pickle=True)
unique_bag = []
for i, CSet in enumerate(CSet_collection):
    new_CSet, unique_CSet = simplify(CSet)
    CSet_collection[i] = new_CSet
    unique_bag.append(unique_CSet)
print(time.time() - tic, 's')
np.save('new_CSet_collection', CSet_collection)

unique_bag = np.concatenate(unique_bag)
unique_bag = np.unique(unique_bag, axis=0)
print(unique_bag.shape)
new_unique_bag, unique_bag = simplify(unique_bag)
np.save('clique_bag.npy', unique_bag)

'''
clique_bag = np.concatenate(CSet_collection)
np.save('clique_bag', clique_bag)
kmeans = KMeans(n_clusters=2000, random_state=0)
kmeans.fit(clique_bag)
centers = kmeans.cluster_centers_
np.save('centers', centers)
'''
print(time.time() - tic, 's')
