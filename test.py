import numpy as np
import pygalmesh

mesh = pygalmesh.remesh_surface(
    "target.stl",
    max_edge_size_at_feature_edges=0.5,
    min_facet_angle=25,
    max_radius_surface_delaunay_ball=0.5,
    max_facet_distance=0.1,
    verbose=False,
)

mesh.write('target_remeshed.stl')
