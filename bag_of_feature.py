import numpy as np
from sklearn.cluster import KMeans


def construct_bof(DLFS_set: np.ndarray):
    size = DLFS_set.shape[0]
    shape = DLFS_set[0].shape

    con_set = []
    for i in range(size):
        DLFS_set[i] = DLFS_set[i].reshape(shape[0], shape[1]*shape[2])
        con_set.append(DLFS_set[i])
        print(con_set[-1].shape)

    con_set = np.asarray(con_set)
    bof = np.concatenate(con_set, axis=0)
    return bof, con_set


def construct_codebook(bof, num_clusters=100):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(bof)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return centers, labels
