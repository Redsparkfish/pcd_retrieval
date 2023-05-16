import numpy as np
from sklearn.cluster import KMeans
import time


tic = time.time()
bof = np.load('C:/Users/Admin/CAD_parts/bof.npy')
print(bof.shape)
num_clusters = 100
choice = np.random.choice(bof.shape[0], 300000, replace=False)
chosen = bof[choice]

kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(chosen)
centers = kmeans.cluster_centers_
labels = kmeans.labels_
np.save('C:/Users/Admin/CAD_parts/codebook.npy', centers)
np.save('C:/Users/Admin/CAD_parts/labels.npy', labels)
print('coded in', time.time()-tic, 'seconds')
