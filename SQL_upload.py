import pymysql
import json
import numpy as np
import os


def numpy_to_json(numpy_array):
    if numpy_array is None:
        return None
    list_array = numpy_array.tolist()
    return json.dumps(list_array)


file = open('configuration.json', 'r')
config = json.load(file)
data_dir = config['data_dir']
meta_path = os.path.join(data_dir, 'meta.npy')
meta = np.load(meta_path, allow_pickle=True)

db = pymysql.connect(host=config['db_host'], user=config['db_user'], password=config['db_password'], database=config['db_database'])
cursor = db.cursor()
size = meta.shape[0]
i = 0
for d in meta:
    data = {'partType': d.get('category'),
            'partName': d.get('name'),
            'distributeDesc': numpy_to_json(d.get('distribution')),
            'bofDesc': numpy_to_json(d.get('bof_desc')),
            'parameterDesc': numpy_to_json(d.get('scale_par'))}
    sql = "INSERT INTO `part_desc` (`partType`, `partName`, `distributeDesc`, `bofDesc`, `parameterDesc`) VALUES (%s, %s, %s, %s, %s) " \
          "AS new ON DUPLICATE KEY UPDATE " \
          "distributeDesc=new.distributeDesc, " \
          "bofDesc=new.bofDesc, " \
          "parameterDesc=new.parameterDesc"
    cursor.execute(sql, (data['partType'], data['partName'], data['distributeDesc'], data['bofDesc'], data['parameterDesc']))

db.commit()
db.close()
