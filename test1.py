import json
import os
import numpy as np
import shutil

with open('meta.json', 'r') as file:
    meta = json.load(file)
    print(meta)
    file.close()

print('aaa'.startswith())