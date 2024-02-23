import os
import shutil
import json


source_dir = r"C:\Users\Admin\CAD_parts"
test_dir = r"C:\Users\Admin\dataset_test"

meta = []

for folder in os.listdir(source_dir):
    if not os.path.isdir(os.path.join(source_dir, folder)):
        continue
    if folder in ['update', 'temp_update', 'delete', 'unclassified']:
        continue
    # if not folder.lower().startswith("b"):
    #     continue
    print(folder)
    for stl_file in os.listdir(os.path.join(source_dir, folder, 'STL')):
        if not stl_file.endswith('.stl'):
            continue
        d = dict()
        d['partName'] = stl_file[:-4]
        d['partType'] = folder
        d['clientInfo'] = 'None'
        d['uploaded'] = "0"
        d['trained'] = "0"
        shutil.copy(os.path.join(source_dir, folder, 'STL', stl_file),
                    os.path.join(test_dir, 'STL', stl_file))
        meta.append(d)
    for stp_file in os.listdir(os.path.join(source_dir, folder, 'STEP')):
        if not stp_file.endswith('.stp'):
            continue
        shutil.copy(os.path.join(source_dir, folder, 'STEP', stp_file),
                    os.path.join(test_dir, 'STP', stp_file))
    for pcd_file in os.listdir(os.path.join(source_dir, folder, 'PCD')):
        if not pcd_file.endswith('.npy'):
            continue
        shutil.copy(os.path.join(source_dir, folder, 'PCD', pcd_file),
                    os.path.join(test_dir, 'PCD', pcd_file))
    for dct_file in os.listdir(os.path.join(source_dir, folder, 'DCT')):
        if not dct_file.endswith('.npy'):
            continue
        shutil.copy(os.path.join(source_dir, folder, 'DCT', dct_file),
                    os.path.join(test_dir, 'DCT', dct_file))

meta = json.dumps(meta, indent=2, ensure_ascii=False)
with open(test_dir+'/meta.json', 'w') as meta_json:
    meta_json.write(meta)
    meta_json.close()
