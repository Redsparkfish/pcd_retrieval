import numpy as np
import json


def numpy_to_json(numpy_array):
    list_array = numpy_array.tolist()
    return json.dumps(list_array)


a = {'name': 'Jack'}
b = np.ones(10)
result_a = json.dumps(a.get('gender'))
result_b = json.dumps(b)
print(result_b)
