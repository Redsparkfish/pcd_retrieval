import json
import os
import numpy as np
import json


def numpy_to_json(numpy_array):
    if numpy_array is None:
        return json.dumps(np.zeros(17).tolist())
    list_array = numpy_array.tolist()
    return json.dumps(list_array)


with open(r'C:\Users\Admin\classified_mesh\meta.json', 'r') as metafile:
    meta = json.load(metafile)

np.save(r'C:\Users\Admin\classified_mesh\meta.npy', meta)

