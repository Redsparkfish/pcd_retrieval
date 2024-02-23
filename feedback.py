import json
import numpy as np
import argparse
import sys


def get_descs(path: str):
    with open(path, 'r', encoding="utf-8") as file:
        desc_json = json.load(file)
    param_descs = np.array([eval(d["paramDesc"]) for d in desc_json])
    bof_descs = np.array([eval(d["bofDesc"]) for d in desc_json])
    d2_descs = np.array([eval(d["d2Desc"]) for d in desc_json])
    return param_descs, bof_descs, d2_descs


parser = argparse.ArgumentParser()
parser.add_argument("--query_path", default="query.json")
parser.add_argument("--config_path", default="configuration.json")
parser.add_argument("--retrieved_path", default="retrieved.json")
parser.add_argument("--relevant_path", default="relevant.json")
parser.add_argument("--irrelevant_path", default="irrelevant.json")
args = parser.parse_args()
with open(args.config_path, 'r', encoding="utf-8") as config_file:
    config = json.load(config_file)
data_dir = config["data_dir"]
meta_path = data_dir + "/meta.json"
with open(meta_path, 'r', encoding="utf-8") as meta_file:
    meta = json.load(meta_file)

retrieved_param, retrieved_bof, retrieved_d2 = get_descs(args.retrieved_path)
retrieved_descs = np.hstack((retrieved_param, retrieved_bof))
relevant_param, relevant_bof, relevant_d2 = get_descs(args.relevant_path)
relevant_descs = np.hstack((relevant_param, relevant_bof))
irrelevant_param, irrelevant_bof, irrelevant_d2 = get_descs(args.irrelevant_path)
irrelevant_descs = np.hstack((irrelevant_param, irrelevant_bof))

param_dim = retrieved_param.shape[1]
bof_dim = retrieved_bof.shape[1]
dim = param_dim+bof_dim
epsilon = 0.0001
r_size = relevant_descs.shape[0]
irr_size = irrelevant_descs.shape[0]
relevant_ranges = np.vstack((np.min(relevant_descs, axis=0), np.max(relevant_descs, axis=0)))
delta = np.ones(dim)
for i in range(dim):
    in_dom = 0
    for j in range(irr_size):
        if (irrelevant_descs[j, i] <= relevant_ranges[1, i]) and (irrelevant_descs[j, i] >= relevant_ranges[0, i]):
            in_dom += 1
    delta[i] -= in_dom / max(1, irr_size)

if (delta[:param_dim] == np.zeros(param_dim)).all():
    delta[:param_dim] = np.ones(param_dim)

if (delta[param_dim:] == np.zeros(bof_dim)).all():
    delta[param_dim:] = np.ones(bof_dim)
retrieved_std = np.std(retrieved_descs, axis=0)
relevant_std = np.std(relevant_descs, axis=0)
weights = delta * (epsilon + retrieved_std) / (epsilon + relevant_std)
weights[:param_dim] /= weights[:param_dim].sum()
weights[:param_dim] *= param_dim
weights[param_dim:] /= weights[param_dim:].sum()
weights[param_dim:] *= bof_dim
param_weights = weights[:param_dim].tolist()
bof_weights = weights[param_dim:].tolist()
weights_json = {"param_weights": param_weights, "bof_weights": bof_weights}
print('[')
print(json.dumps(weights_json, indent=2))

new_query = {}

try:
    with open(args.query_path, mode='r', encoding="utf-8") as query_json:
        query = json.load(query_json)
    if "paramDesc" in query.keys():
        query_param = np.array(eval(query.get("paramDesc")))
        new_param = query_param + np.sum(relevant_param, axis=0) / r_size - np.sum(irrelevant_param, axis=0) / max(1, irr_size)
        new_param *= np.sum(query_param) / np.sum(new_param)
        new_query["paramDesc"] = new_param.tolist()
    if "bofDesc" in query.keys():
        query_bof = np.array(eval(query.get("bofDesc")))
        new_bof = query_bof + np.sum(relevant_bof, axis=0) / r_size - np.sum(irrelevant_bof, axis=0) / max(1, irr_size)
        new_bof *= np.sum(query_bof) / np.sum(new_bof)
        new_query["bofDesc"] = new_bof.tolist()
    if "d2Desc" in query.keys():
        query_d2 = np.array(eval(query.get("d2Desc")))
        new_d2 = query_d2 + np.sum(relevant_d2, axis=0) / r_size - np.sum(irrelevant_d2, axis=0) / max(1, irr_size)
        new_d2 *= np.sum(query_d2) / np.sum(new_d2)
        new_query["d2Desc"] = new_d2.tolist()
except:
    print(']')
    print("Failed to read the query.")
    sys.exit()

print(',')
print(json.dumps(new_query, indent=2))
print(']')
