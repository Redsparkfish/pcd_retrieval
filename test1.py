import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

f = open('configuration.json', mode='r', encoding='utf-8')
config = json.load(f)
data_dir = config['data_dir']
meta = np.load(os.path.join(data_dir, 'meta.npy'), allow_pickle=True)
print(meta)
