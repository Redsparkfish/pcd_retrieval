from global_descriptor import get_par, read_step_file
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stp_path', '-stpp', type=str, default=r"C:\Users\Admin\dataset_test\STP\Bearings_00ed2536-3d80-4f07-8851-4f49f1606498.stp")
    args = parser.parse_args()
    par, scale_par = get_par(read_step_file(args.stp_path))
    output = {"paramDesc": scale_par.tolist()}
    print(json.dumps(output))
