import json
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', '-cp', type=str, default=r"configuration.json")
args = parser.parse_args()
with open(args.config_path, encoding='utf-8') as config_file:
    config = json.load(config_file)

data_dir = config['data_dir']

meta_path = config['data_dir'] + '/meta.json'
with open(meta_path, 'r', encoding='utf-8') as file:
    meta = json.load(file)

new_meta_path = config['data_dir'] + '/new/meta.json'
with open(new_meta_path, 'r', encoding='utf-8') as file:
    new_meta = json.load(file)
if isinstance(new_meta, dict):
    new_meta = [new_meta]

existing_names = set([d['partName'] for d in meta])
new_names = [d['partName'] for d in new_meta]
for i, name in enumerate(new_names):
    stl_src = data_dir + '/new/STL/' + name + '.stl'
    stp_src = data_dir + '/new/STP/' + name + '.stp'
    if not os.path.exists(stl_src) or not os.path.exists(stp_src):
        continue
    if name in existing_names:
        number = 1
        new_name = name + str(number)
        other_new_names = new_names[:i] + new_names[i+1:]
        while new_name in existing_names or new_name in other_new_names:
            number += 1
            new_name = name + str(number)
        new_meta[i]['partName'] = new_name
        new_names[i] = new_name
        stl_tgt = data_dir + '/new/STL/' + new_name + '.stl'
        stp_tgt = data_dir + '/new/STP/' + new_name + '.stp'
        if os.path.exists(stl_src):
            os.rename(stl_src, stl_tgt)
        if os.path.exists(stp_src):
            os.rename(stp_src, stp_tgt)


for d in new_meta:
    stl_src = data_dir + '/new/STL/' + d['partName'] + '.stl'
    stp_src = data_dir + '/new/STP/' + d['partName'] + '.stp'
    if not os.path.exists(stl_src):
        print("stl file for", d['partName'], 'not found.')
        continue
    if not os.path.exists(stp_src):
        print('stp file for', d['partName'], 'not found')
        continue
    meta.append(d)

STL_src = data_dir + '/new/STL'
STL_tgt = data_dir + '/STL'
STP_src = data_dir + '/new/STP'
STP_tgt = data_dir + '/STP'

if os.path.exists(STL_src):
    shutil.copytree(STL_src, STL_tgt, dirs_exist_ok=True)
    shutil.rmtree(STL_src)
    os.mkdir(STL_src)
if os.path.exists(STP_src):
    shutil.copytree(STP_src, STP_tgt, dirs_exist_ok=True)
    shutil.rmtree(STP_src)
    os.mkdir(STP_src)

meta = json.dumps(meta, indent=2, ensure_ascii=False)
with open(meta_path, 'w', encoding='utf-8') as meta_json:
    meta_json.write(meta)
print('Uploading finished.')
