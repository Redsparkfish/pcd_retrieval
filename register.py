import trimesh
import time
import numpy as np


mesh1 = trimesh.load_mesh(r'C:\Users\Admin\CAD_assemblies\CouplingFlange/Unnamed-flange-coupling-022.obj')
print(mesh1.edges)
labels = trimesh.graph.connected_component_labels(mesh1.face_adjacency)
print(labels.max())