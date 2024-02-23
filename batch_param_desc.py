import argparse
import json
import os
from global_descriptor import get_par, read_step_file
import numpy as np


def batch_param_desc(meta):
    for d in meta:
        if d.get('paramDesc') and eval(d['paramDesc']) != np.zeros(17).tolist():
            print(d["partName"], 'paramDesc exists.')
            continue
        stp_path = os.path.join(data_dir, 'STP', d.get('partName') + '.stp')
        try:
            print("Start calculating paramDesc for", d.get('partName'))
            par, scale_par = get_par(read_step_file(stp_path))
        except:
            print("STP parameters calculation failed, using zeros instead...")
            scale_par = np.zeros(17)
        d["paramDesc"] = json.dumps(scale_par.tolist())
        d['uploaded'] = "0"
        print(d.get('partName'), "'s paramDesc is calculated")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-cp', type=str, default=r"configuration.json")
    args = parser.parse_args()
    with open(args.config_path, 'r', encoding="utf-8") as config_file:
        config = json.load(config_file)
        config_file.close()
    data_dir = config["data_dir"]
    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path, 'r', encoding="utf-8") as meta_json:
        meta = json.load(meta_json)
        meta_json.close()
    batch_param_desc(meta)
    with open(meta_path, 'w', encoding="utf-8") as meta_json:
        meta = json.dumps(meta, indent=2, ensure_ascii=False)
        meta_json.write(meta)