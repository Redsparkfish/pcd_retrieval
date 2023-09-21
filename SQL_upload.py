import pymysql
import json
import numpy as np
import os


def numpy_to_json(numpy_array):
    if numpy_array is None:
        return json.dumps(np.zeros(17).tolist())
    list_array = numpy_array.tolist()
    return json.dumps(list_array)


def SQL_upload(meta, config):
    db = pymysql.connect(host=config['db_host'], user=config['db_user'], password=config['db_password'],
                         database=config['db_database'])
    cursor = db.cursor()
    i = 0
    for d in meta:
        data = {'partType': d.get('partType'),
                'partName': d.get('partName'),
                'clientInfo': "none",
                'd2Desc': json.dumps(d.get('d2_desc')),
                'bofDesc': json.dumps(d.get('bof_desc')),
                'paramDesc': json.dumps(d.get('param_desc'))}
        sql = "INSERT INTO `part` (`partType`, `partName`, `clientInfo`, `d2Desc`, `bofDesc`, `paramDesc`) VALUES " \
              "(%s, %s, %s, %s, %s, %s) " \
              "AS new ON DUPLICATE KEY UPDATE " \
              "d2Desc=new.d2Desc, " \
              "bofDesc=new.bofDesc, " \
              "paramDesc=new.paramDesc"
        cursor.execute(sql, (
        data['partType'], data['partName'], data['clientInfo'], data['d2Desc'], data['bofDesc'], data['paramDesc']))

    db.commit()
    db.close()


if __name__ == "__main__":
    pass
    # file = open('configuration.json', 'r')
    # config = json.load(file)
    # data_dir = config['data_dir']
    # meta_path = os.path.join(data_dir, 'meta.json')
    # with open(meta_path, 'r') as metafile:
    #     meta = json.load(metafile)
    # SQL_upload(meta, config)
