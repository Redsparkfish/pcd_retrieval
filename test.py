import numpy as np
import json
import os
from retrieval_test import retrieve_test

with open('configuration.json', 'r') as file:
    config = json.load(file)
    file.close()
data_dir = config['data_dir']

with open(os.path.join(data_dir, 'meta1.json'), 'r') as meta_file:
    meta = json.load(meta_file)
    meta_file.close()

summary = 0
k = 10
batch_descs = np.array([d['desc'] for d in meta])
size = batch_descs.shape[0]

for query in meta:
    query_name = query['partName']
    query_type = query['partType']
    query_desc = np.array(query['desc'])

    results_idx = retrieve_test(query_desc, batch_descs, k)
    for i in results_idx[1:]:
        if meta[i]['partType'] == query_type:
            summary += 1
        elif query_type in ['Rotary_Shaft', 'Keyway_Shaft'] and meta[i]['partType'] in ['Rotary_Shaft', 'Keyway_Shaft']:
            summary += 1

print(summary / (size*k))

