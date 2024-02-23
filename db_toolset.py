import argparse
import pymysql
import json
from DLFS_calculation import *
from bag_of_feature import *
from global_descriptor import *


group_size = 200
num_clusters_per_group = 100


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


def SQL_upload(meta, config):
    db = pymysql.connect(host=config['db_host'], user=config['db_user'], password=config['db_password'],
                         database=config['db_database'])
    cursor = db.cursor()
    i = 0
    for d in meta:
        if d.get('uploaded') and mode != 'initialize':
            continue
        print('uploading', d.get('partName'))
        columns = ', '.join([column for column in d.keys() if column not in ['uploaded', 'trained']])
        placeholders = ', '.join(['%s'] * (len(d)-2))
        update_clause = ', '.join([f"{column} = %s" for column in d.keys() if column not in ['uploaded', 'trained']])
        sql = f"INSERT INTO {config['db_table']} ({columns}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_clause}"

        values = tuple([d[column] for column in d.keys() if column not in ['uploaded', 'trained']])
        cursor.execute(sql, values * 2)
        d['uploaded'] = True
        i += 1

    db.commit()
    cursor.close()
    db.close()
    print("Uploading has finished.")


def batch_DLFS(meta: list):
    tic = time.time()
    DLFS_path_list = []
    if not os.path.exists(os.path.join(data_dir, 'DCT')):
        os.makedirs(os.path.join(data_dir, 'DCT'))

    if not os.path.exists(os.path.join(data_dir, 'PCD')):
        os.makedirs(os.path.join(data_dir, 'PCD'))

    for d in meta:
        partName = d['partName']
        DLFS_path = os.path.join(data_dir, 'DCT', partName + '.npy')
        if os.path.exists(DLFS_path):
            if mode == 'initialize':
                DLFS_path_list.append(DLFS_path)
            continue
        stl_path = os.path.join(data_dir, 'STL', partName + '.stl')
        try:
            mesh = trimesh.load(stl_path, force='mesh')
            points = trimesh.sample.sample_surface(mesh, 50000)[0]
        except:
            print("DLFS calculation for", stl_path, "failed")
            continue
        DLFS_path_list.append(DLFS_path)
        points = np.unique(points, axis=0)
        np.save(os.path.join(data_dir, 'PCD', partName + '.npy'), points)
        mr = meshResolution(points)

        # Extract ISS keypoints
        print(partName)
        key_indices, neighbor_indices, eigvectors = computeISS(points, rate=1, radius=2.5 * mr)
        # Extract DLFS features from the mesh
        LMA = getLMA(points, eigvectors, radius=7 * mr)
        DLFSs = getDLFS(points, LMA, key_indices, R=20 * mr)
        DLFSs = DLFSs.reshape(DLFSs.shape[0], DLFSs.shape[1] * DLFSs.shape[2])
        # Append the features to the list
        np.save(DLFS_path, DLFSs)

    print('DLFS calculation finished in', (time.time() - tic) / 60, 'min.')
    return DLFS_path_list


def get_kmeans_list(DLFS_path_list: list):
    size = len(DLFS_path_list)
    current_group = 0
    new_kmeans_list = []
    while size >= group_size:
        DLFS_set = []
        for i in range(group_size):
            idx = group_size * current_group + i
            DLFS_set.append(np.load(DLFS_path_list[idx]))
        bof = np.concatenate(DLFS_set)
        if size > group_size:
            print('Start Kmeans clustering for group', current_group)
        else:
            print('Start Kmeans clustering for the last group.')
        kmeans = construct_codebook(bof, num_clusters=num_clusters_per_group)
        new_kmeans_list.append(kmeans)
        size -= group_size
        current_group += 1
    residual_path_list = []
    if size != 0:
        DLFS_set = []
        for i in range(size):
            idx = group_size * current_group + i
            DLFS_set.append(np.load(DLFS_path_list[idx]))
            residual_path_list.append(DLFS_path_list[idx])
        bof = np.concatenate(DLFS_set)
        print('Start Kmeans clustering for the last group.')
        kmeans = construct_codebook(bof, num_clusters=num_clusters_per_group)
        new_kmeans_list.append(kmeans)
    return new_kmeans_list, residual_path_list


def train(meta, residual_list=[], old_kmeans_list=[]):
    tic0 = time.time()
    DLFS_path_list = []
    for d in meta:
        if d.get('trained') is False:
            DLFS_path_list.append(os.path.join(data_dir, 'DCT', d['partName'] + '.npy'))
            d['trained'] = True
    DLFS_path_list = residual_list + DLFS_path_list
    kmeans_list, residual_path_list = get_kmeans_list(DLFS_path_list)
    kmeans_list = old_kmeans_list + kmeans_list
    if len(kmeans_list) < 2:
        high_kmeans = kmeans_list[0]
    else:
        high_kmeans = construct_high_codebook(kmeans_list)
    print('Training finished in', (time.time() - tic0) / 60, 'min. Initializing descriptor calculation.')
    return residual_path_list, kmeans_list, high_kmeans


def batch_pcd_desc(meta, kmeans_list, high_kmeans):
    for d in meta:
        if d.get('uploaded') is True and mode.lower() == 'upload_new':
            continue
        DLFS_path = os.path.join(data_dir, 'DCT', d.get('partName') + '.npy')
        PCD_path = os.path.join(data_dir, 'PCD', d.get('partName') + '.npy')
        try:
            DLFS = np.load(DLFS_path)
        except:
            print("DLFS loading for", d.get('partName'), "failed")
            continue
        points = np.load(PCD_path)
        d2_desc = D2(points).tolist()
        bof_desc = sparseCoding(DLFS, high_kmeans, kmeans_list).tolist()
        d['d2Desc'] = json.dumps(d2_desc)
        d['bofDesc'] = json.dumps(bof_desc)
        d['uploaded'] = False
        print(d.get('partName'))
    return meta


def batch_param_desc(meta):
    for d in meta:
        if d.get('uploaded') is True:
            continue
        stp_path = os.path.join(data_dir, 'STP', d.get('partName') + '.stp')
        try:
            par, scale_par = get_par(read_step_file(stp_path))
        except:
            print("STP parameters calculation failed, using zeros instead...")
            scale_par = np.zeros(17)
        d["paramDesc"] = json.dumps(scale_par.tolist())
        d['uploaded'] = False
        print(d.get('partName'))
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-cp', type=str, default="configuration.json")
    args = parser.parse_args()
    with open(args.config_path, 'r', encoding="utf-8") as config_file:
        config = json.load(config_file)
        config_file.close()
    data_dir = config["data_dir"]
    server_ip = config["server_ip"]
    server_port = config["server_port"]
    url = "http://" + server_ip + ":" + server_port + "/api/op/synchronize-database"
    mode = config["mode"].lower()
    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'r') as meta_json:
        meta = json.load(meta_json)
        meta_json.close()
    try:
        if mode == 'initialize':
            DLFS_path_list = batch_DLFS(meta)
            residual_path_list, kmeans_list, high_kmeans = train(meta)
            np.save(os.path.join(data_dir, "residual_path_list.npy"), residual_path_list)
            np.save(os.path.join(data_dir, "kmeans_list.npy"), kmeans_list)
            np.save(os.path.join(data_dir, "high_kmeans.npy"), [high_kmeans])
            # kmeans_list = np.load(os.path.join(data_dir, "kmeans_list.npy"), allow_pickle=True).tolist()
            # high_kmeans = np.load(os.path.join(data_dir, "high_kmeans.npy"), allow_pickle=True)[0]

            meta = batch_pcd_desc(meta, kmeans_list, high_kmeans)
            meta = batch_param_desc(meta)
            with open(meta_path, 'w') as meta_json:
                meta = json.dumps(meta, indent=2, ensure_ascii=False)
                meta_json.write(meta)
                meta = json.loads(meta)
                meta_json.close()
            SQL_upload(meta, config)
        elif mode == 'upload_new':
            DLFS_path_list = batch_DLFS(meta)

            kmeans_list = np.load(os.path.join(data_dir, "kmeans_list.npy"), allow_pickle=True).tolist()
            high_kmeans = np.load(os.path.join(data_dir, "high_kmeans.npy"), allow_pickle=True)[0]

            meta = batch_pcd_desc(meta, kmeans_list, high_kmeans)
            meta = batch_param_desc(meta)
            SQL_upload(meta, config)
        elif mode == 'update_model':
            DLFS_path_list = np.load(os.path.join(data_dir, 'residual_path_list.npy'), allow_pickle=True).tolist()
            old_kmeans_list = np.load(os.path.join(data_dir, "kmeans_list.npy"), allow_pickle=True).tolist()
            if len(DLFS_path_list) != 0:
                old_kmeans_list = old_kmeans_list[:-1]

            residual_path_list, kmeans_list, high_kmeans = train(meta, residual_list=DLFS_path_list, old_kmeans_list=old_kmeans_list)
            kmeans_list = old_kmeans_list + kmeans_list
            if len(kmeans_list) < 2:
                high_kmeans = kmeans_list[0]
            else:
                high_kmeans = construct_high_codebook(kmeans_list)
            np.save(os.path.join(data_dir, "residual_path_list.npy"), residual_path_list)
            np.save(os.path.join(data_dir, "kmeans_list.npy"), kmeans_list)
            np.save(os.path.join(data_dir, "high_kmeans.npy"), [high_kmeans])

            meta = batch_pcd_desc(meta, kmeans_list, high_kmeans)
            with open(meta_path, 'w') as meta_json:
                meta = json.dumps(meta, indent=2, ensure_ascii=False)
                meta_json.write(meta)
                meta = json.loads(meta)
                meta_json.close()
            SQL_upload(meta, config)
        elif mode == 'reset':
            for d in meta:
                d['uploaded'] = False
                d['trained'] = False
                if 'd2Desc' in d.keys():
                    d.pop('d2Desc')
                    d.pop('bofDesc')
                if 'paramDesc' in d.keys():
                    d.pop('paramDesc')
            os.remove(os.path.join(data_dir, "residual_path_list.npy"))
            os.remove(os.path.join(data_dir, "kmeans_list.npy"))
            os.remove(os.path.join(data_dir, "high_kmeans.npy"))

        else:
            raise "Invalid mode error. "

        with open(meta_path, 'w') as meta_json:
            meta = json.dumps(meta, indent=2, ensure_ascii=False)
            meta_json.write(meta)
            meta_json.close()
        print("Process finished successfully.")
    except Exception as e:
        print(e)
