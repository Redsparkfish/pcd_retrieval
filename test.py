import numpy as np
import json
import os

with open('configuration.json', 'r') as file:
    config = json.load(file)
data_dir = config['data_dir']
names_list = np.load(os.path.join(data_dir, 'names_list.npy'), allow_pickle=True).tolist()

a = [[1, 2, 3], [3, 4]]
a[0].pop(1)
print(a)