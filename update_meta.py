import json
import os


with open('configuration.json', 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)
print('configuration is successfully read.')
data_dir = config['data_dir']
meta_path = data_dir + '/meta.json'

with open(meta_path, 'r', encoding='utf-8') as file:
    meta = json.load(file)
existing_names = set()
for d in meta:
    existing_names.add(d['partName'])

partName_list = []
for stl in os.listdir(data_dir+'/STL'):
    if not stl.endswith('.stl'):
        continue
    name = stl[:-4]
    if name not in existing_names:
        partName_list.append(name)

for name in partName_list:
    d = {
        "partName": name,
        "trained": "0",
        "uploaded": "0"
    }
    meta.append(d)
    print(name, "added to meta")

meta = json.dumps(meta, indent=2, ensure_ascii=False)
with open(meta_path, 'w', encoding='utf-8') as meta_json:
    meta_json.write(meta)
    meta_json.close()
