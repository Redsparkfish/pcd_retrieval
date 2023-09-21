import argparse
import shutil
import time
from DLFS_calculation import computeDLFS
from SQL_upload import *
from bag_of_feature import *
from global_descriptor import *

batch_size = 500
num_clusters_batch = 100


def open_json(json_file, mode='r'):
    with open(json_file, mode=mode) as file:
        result = json.load(file)
        file.close()
    return result


def get_kmeans_list(data_dir, train_partTypes, train_partNames):
    size = train_partNames.shape[0]
    current_batch = 0
    new_kmeans_list = []
    while size >= batch_size:
        DLFS_set = []
        for i in range(batch_size):
            idx = batch_size * current_batch + i
            DLFS_set.append(np.load(os.path.join(data_dir, train_partTypes[idx], 'DCT', train_partNames[idx] + '.npy')))
        bof = np.concatenate(DLFS_set)
        print('Start Kmeans clustering.')
        kmeans = construct_codebook(bof, num_clusters=num_clusters_batch)
        new_kmeans_list.append(kmeans)
        size -= batch_size
        current_batch += 1

    if size != 0:
        DLFS_set = []
        for i in range(size):
            idx = batch_size * current_batch + i
            DLFS_set.append(np.load(os.path.join(data_dir, train_partTypes[idx], 'DCT', train_partNames[idx] + '.npy')))
        bof = np.concatenate(DLFS_set)
        print('Start Kmeans clustering.')
        kmeans = construct_codebook(bof, num_clusters=num_clusters_batch)
        new_kmeans_list.append(kmeans)
    return new_kmeans_list


def meta_from_SQL(config):
    meta = []
    db = pymysql.connect(host=config['db_host'], user=config['db_user'], password=config['db_password'],
                         database=config['db_database'])
    cursor = db.cursor()
    sql_query = "SELECT * FROM part"
    cursor.execute(sql_query)

    column_names = [column[0] for column in cursor.description]
    result = cursor.fetchall()

    for record in result:
        d = {}
        for i, name in enumerate(column_names):
            d[name] = record[i]
        meta.append(d)
    return meta


def get_residuals(train_partNames, train_partTypes):
    while len(train_partTypes) >= batch_size:
        train_partTypes = train_partTypes[batch_size:]
        train_partNames = train_partNames[batch_size:]

    residuals = []
    for i in range(len(train_partTypes)):
        residuals.append({'partType': train_partTypes[i],
                          'partName': train_partNames[i]})

    return residuals


def train(data_dir, train_partNames, train_partTypes, old_kmeans_list=None):
    tic0 = time.time()
    kmeans_list = get_kmeans_list(data_dir, train_partTypes, train_partNames)
    if old_kmeans_list:
        kmeans_list = old_kmeans_list + kmeans_list
    residuals = get_residuals(train_partNames, train_partTypes)

    if len(kmeans_list) < 2:
        high_kmeans = kmeans_list[0]
    else:
        high_kmeans = construct_high_codebook(kmeans_list)
    print('Training finished in', (time.time() - tic0) / 60, 'min. Initializing descriptor calculation.')
    return kmeans_list, high_kmeans, residuals


def calc_param(meta):
    bug_file = open(os.path.join(config["data_dir"], 'bug_log.txt'), 'a')
    tic = time.time()
    for d in meta:
        category = d['partType']
        name = d['partName']
        if d.get('param_desc') and d['param_desc'] != np.zeros(17).tolist():
            print(name, 'param_desc exists.')
            continue
        try:
            scale_par = param_desc(config["data_dir"], category, name)
            print(name, 'param_desc calculated.')
        except:
            bug_file.write(f'scale param for {name} failed.\n')
            scale_par = np.zeros(17, dtype=float).tolist()
            print(name, 'param_desc calculation failed. Zeros are applied.')
        d['param_desc'] = scale_par.tolist()
    bug_file.close()
    print('param_descs calculation finished in', (time.time() - tic) / 60, 'min.')
    return meta


def start(config):
    data_dir = config["data_dir"]
    key_indices_list, new_partNames, new_partTypes = computeDLFS(data_dir, config["mode"])
    train_partNames = new_partNames.copy()
    train_partTypes = new_partTypes.copy()
    kmeans_list, high_kmeans, residuals = train(data_dir, train_partNames, train_partTypes)

    np.save(os.path.join(data_dir, 'kmeans_list.npy'), kmeans_list)
    np.save(os.path.join(data_dir, 'high_kmeans.npy'), [high_kmeans])
    with open(os.path.join(data_dir, 'residuals.json'), 'w') as outfile:
        residuals = json.dumps(residuals, indent=2, ensure_ascii=False)
        outfile.write(residuals)
        outfile.close()

    meta = computeGlobalDescriptors(data_dir, high_kmeans, kmeans_list, new_partTypes, new_partNames)
    print("Initializing param_desc calculation...")
    meta = calc_param(meta)
    SQL_upload(meta, config)


def update(config):
    data_dir = config["data_dir"]
    old_meta = meta_from_SQL(config)
    old_kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True).tolist()
    old_partTypes = [d['partType'] for d in old_meta]
    old_partNames = [d['partName'] for d in old_meta]

    residuals_path = os.path.join(data_dir, 'residuals.json')
    if os.path.exists(residuals_path):
        residuals = open_json(residuals_path)
    else:
        residuals= get_residuals(old_partNames, old_partTypes)
    if len(residuals) > 0:
        old_kmeans_list = old_kmeans_list[:-1]
    residual_partTypes = [d['partType'] for d in residuals]
    residual_partNames = [d['partName'] for d in residuals]

    update_dir = os.path.join(data_dir, 'update')
    key_indices_list, new_partNames, new_partTypes = computeDLFS(update_dir, config["mode"])
    train_partNames = np.append(residual_partNames, new_partNames)
    train_partTypes = np.append(residual_partTypes, new_partTypes)

    last_id = old_meta[-1].get('id') + 1

    for partType in np.unique(new_partTypes):
        src = os.path.join(update_dir, partType)
        dest = os.path.join(data_dir, partType)
        shutil.copytree(src, dest, dirs_exist_ok=True)
        shutil.rmtree(src)
        os.mkdir(src)
        os.mkdir(src+'/STL')
        os.mkdir(src+'/STP')

    kmeans_list, high_kmeans, residuals = train(data_dir, train_partNames, train_partTypes, old_kmeans_list)
    np.save(os.path.join(data_dir, 'kmeans_list.npy'), kmeans_list)
    np.save(os.path.join(data_dir, 'high_kmeans.npy'), [high_kmeans])
    with open(os.path.join(data_dir, 'residuals.json'), 'w') as outfile:
        residuals = json.dumps(residuals, indent=2, ensure_ascii=False)
        outfile.write(residuals)
        outfile.close()

    new_meta = computeGlobalDescriptors(data_dir, high_kmeans, kmeans_list, new_partTypes, new_partNames)
    print("Initializing param_desc calculation...")
    new_meta = calc_param(new_meta)
    SQL_upload(new_meta, config)


def temp_update(config):
    data_dir = config['data_dir']
    update_dir = os.path.join(data_dir, "temp_update")

    kmeans_list = np.load(os.path.join(data_dir, 'kmeans_list.npy'), allow_pickle=True).tolist()
    high_kmeans = np.load(os.path.join(data_dir, 'high_kmeans.npy'), allow_pickle=True)[0]

    key_indices_list, new_partNames, new_partTypes = computeDLFS(update_dir, config['mode'])
    new_meta = computeGlobalDescriptors(update_dir, high_kmeans, kmeans_list, new_partTypes, new_partNames)

    for partType in np.unique(new_partTypes):
        src = os.path.join(update_dir, partType)
        dest1 = os.path.join(data_dir, 'update', partType)
        dest2 = os.path.join(data_dir, partType)
        shutil.copytree(src, dest1, dirs_exist_ok=True)
        shutil.copytree(src, dest2, dirs_exist_ok=True)
        shutil.rmtree(src)
        os.mkdir(src)
        os.mkdir(src + '/STL')
        os.mkdir(src + '/STP')

    SQL_upload(new_meta, config)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-cp', type=str, default="configuration.json")
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'r', encoding="utf-8"))

