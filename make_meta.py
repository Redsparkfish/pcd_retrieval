import json


with open('configuration.json', 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

meta_path = config['data_dir'] + '/meta.json'
with open(meta_path, 'r', encoding='utf-8') as file:
    meta = json.load(file)

for d in meta:
    keys = list(d.keys())
    for key in keys:
        if d[key] == "":
            d.pop(key)
    if 'precision' in keys:
        d.pop('precision')

meta = json.dumps(meta, indent=2, ensure_ascii=False)
with open(meta_path, 'w', encoding='utf-8') as meta_json:
    meta_json.write(meta)
    meta_json.close()