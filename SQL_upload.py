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
names = np.load(os.path.join(data_dir, 'names.npy'))
categories = np.load(os.path.join(data_dir, 'categories.npy'))
d2_list = np.load(os.path.join(data_dir, 'd2_list.npy'))
bof_desc_list = np.load(os.path.join(data_dir, 'bof_Desc.npy'))

db = pymysql.connect(host=config['db_host'], user=config['db_user'], password=config['db_password'], database=config['db_database'])
cursor = db.cursor()
size = names.shape[0]
i = 0
for i in range(size):
    data = {'partType': categories[i],
            'partName': names[i],
            'distributeDesc': numpy_to_json(d2_list[i]),
            'bofDesc': numpy_to_json(bof_desc_list[i])}
    sql = "INSERT INTO `part_desc` (`partType`, `partName`, `distributeDesc`, `bofDesc`) VALUES (%s, %s, %s, %s) " \
          "AS new ON DUPLICATE KEY UPDATE " \
          "distributeDesc=new.distributeDesc, " \
          "bofDesc=new.bofDesc"
    cursor.execute(sql, (data['partType'], data['partName'], data['distributeDesc'], data['bofDesc']))

db.commit()
db.close()
