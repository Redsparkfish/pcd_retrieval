import open3d as o3d
import numpy as np


data_dir = 'c:/users/admin/modelnet40_normal_resampled/'
file = data_dir + 'bathtub/bathtub_0024.txt'
file1 = data_dir + 'desk/desk_0018.txt'
points = np.loadtxt(file, delimiter=',')[:, :3]
points1 = np.loadtxt(file1, delimiter=',')[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points1)

o3d.visualization.draw([pcd])
o3d.visualization.draw([pcd1])
