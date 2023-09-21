from DLFS_calculation import *
from global_descriptor import D2, sparseCoding, param_desc
import argparse
import json


def calc_mesh_descs(mesh, high_kmeans, kmeans_list):

    points = trimesh.sample.sample_surface(mesh, 30000)[0]
    mr = meshResolution(points) / 1.25
    tic00 = time.time()
    key_indices, neighbor_indices, eigvectors = computeISS(points, rate=1, radius=2.5 * mr)
    tic01 = time.time()
    LMA = getLMA(points, eigvectors, radius=7 * mr)
    DLFSs = getDLFS(points, LMA, key_indices, R=20 * mr)
    tic02 = time.time()
    DLFSs = DLFSs.reshape(DLFSs.shape[0], DLFSs.shape[1]*DLFSs.shape[2])
    d2_desc = D2(points)
    bof_desc = sparseCoding(DLFSs, high_kmeans, kmeans_list)
    tic03 = time.time()

    return d2_desc, bof_desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Required Files Paths")
    parser.add_argument('--param_desc', type=str, help="Parameter descriptor.")
    parser.add_argument('--stp_path', type=str, help="The path of the stp file.",
                        default=r"C:\Users\Admin\Bearings_0.stp")
    parser.add_argument('--stl_path', type=str, help="The path of the stl file.", required=True,
                        default=r"C:\Users\Admin\Bearings_0.stl")
    parser.add_argument('--high_kmeans', type=str, help="The path of high_kmeans.npy.", required=True)
    parser.add_argument('--kmeans_list', type=str, help="The path of kmeans_list.npy.", required=True)

    args = parser.parse_args()
    tic = time.time()
    if args.param_desc:
        param_desc = eval(args.param_desc)
    elif args.stp_path:
        param_desc = param_desc(None, None, None, path=args.stp_path)
    else:
        raise "Neither stp_path nor param_desc is passed."
    print("param_desc calculation took", time.time() - tic, 'seconds')
    tic = time.time()
    mesh = trimesh.load_mesh(args.stl_path)
    high_kmeans = np.load(args.high_kmeans, allow_pickle=True)[0]
    kmeans_list = np.load(args.kmeans_list, allow_pickle=True)
    d2_desc, bof_desc = calc_mesh_descs(mesh, high_kmeans, kmeans_list)
    results = {"param_desc": list(param_desc),
               "bof_desc": list(bof_desc),
               "d2_desc": list(d2_desc)}
    print("bof_desc and d2_desc calculation took", time.time() - tic, 'seconds')
    print(json.dumps(results))

