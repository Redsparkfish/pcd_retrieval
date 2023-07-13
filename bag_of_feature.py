import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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
    kmeans.cluster_centers_
    return kmeans


def construct_high_codebook(kmeans_list, num_high_clusters=100):
    all_centers = np.concatenate([kmeans_list[i].cluster_centers_ for i in range(len(kmeans_list))])
    kmeans = KMeans(n_clusters=num_high_clusters, random_state=0)
    kmeans.fit(all_centers)
    return kmeans


def GMM_coding(bof, K=100):
    gm = GaussianMixture(n_components=K, covariance_type='diag', max_iter=10000, random_state=0)
    gm.fit(bof)
    GMM_means = gm.means_
    GMM_cov = gm.covariances_
    GMM_weights = gm.weights_
    return GMM_means, GMM_cov, GMM_weights
