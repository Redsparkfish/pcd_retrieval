import pymysql
import json
import numpy as np
import os


def numpy_to_json(numpy_array):
    list_array = numpy_array.tolist()
    return json.dumps(list_array)


file = open('configuration.json', 'r')
config = json.load(file)
data_dir = config['data_dir']
names = np.load(data_dir+'/names.npy')
d2_list = np.load(data_dir+'/d2_list.npy')
bof_desc_list = np.load(data_dir+'/bof_Desc.npy')

db = pymysql.connect(host=config['db_host'], user=config['db_user'], password=config['db_password'], database=config['db_database'])
cursor = db.cursor()
size = len(list(names))
i = 0
for category in os.listdir(data_dir):
    if not os.path.isdir(os.path.join(data_dir, category)):
        continue
    for file in os.listdir(os.path.join(data_dir, category, 'STL')):
        if not file.endswith('.stl'):
            continue
        data = {'partType': category,
                'partName': file,
                'distributeDesc': numpy_to_json(d2_list[i]),
                'bofDesc': numpy_to_json(bof_desc_list[i])}
        sql = "INSERT INTO `part_desc` (`partType`, `partName`, `distributeDesc`, `bofDesc`) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (data['partType'], data['partName'], data['distributeDesc'], data['bofDesc']))
        i += 1

db.commit()
db.close()
