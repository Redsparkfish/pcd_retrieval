import OCC.Core.TopoDS
import os
import numpy as np
from OCC.Core.Bnd import Bnd_OBB
from OCC.Core.gp import *
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.BRepBndLib import brepbndlib_AddOBB
from OCC.Core.BRepPrimAPI import *
from OCC.Core.BRepBuilderAPI import *
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopAbs import *
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape


def min_dist(s1, s2):
    dss = BRepExtrema_DistShapeShape()
    dss.LoadS1(s1)
    dss.LoadS2(s2)
    dss.Perform()
    return dss.Value()


def convert_bnd_to_shape(the_box):
    barycenter = the_box.Center()
    x_dir = the_box.XDirection()
    y_dir = the_box.YDirection()
    z_dir = the_box.ZDirection()
    half_x = the_box.XHSize()
    half_y = the_box.YHSize()
    half_z = the_box.ZHSize()

    x_vec = gp_XYZ(x_dir.X(), x_dir.Y(), x_dir.Z())
    y_vec = gp_XYZ(y_dir.X(), y_dir.Y(), y_dir.Z())
    z_vec = gp_XYZ(z_dir.X(), z_dir.Y(), z_dir.Z())
    point = gp_Pnt(barycenter.X(), barycenter.Y(), barycenter.Z())
    axes = gp_Ax2(point, gp_Dir(z_dir), gp_Dir(x_dir))
    axes.SetLocation(
        gp_Pnt(point.XYZ() - x_vec * half_x - y_vec * half_y - z_vec * half_z)
    )
    box = BRepPrimAPI_MakeBox(axes, 2.0 * half_x, 2.0 * half_y, 2.0 * half_z).Shape()
    return box


def count_shape(explorer: TopExp_Explorer):
    if not explorer.Value():
        return 0
    count = 1
    while explorer.More():
        count += 1
        explorer.Next()
    explorer.ReInit()
    return count


def get_par(shape: OCC.Core.TopoDS.TopoDS_Solid):
    par = np.zeros(17, dtype=float)
    par_scale = np.zeros(17, dtype=float)

    obb = Bnd_OBB()
    brepbndlib_AddOBB(shape, obb)
    center = gp_Pnt(obb.Center())
    xyz = np.sort(np.array([obb.XHSize(), obb.YHSize(), obb.ZHSize()]))
    par[:3] = xyz/np.sum(xyz)
    par_scale[:3] = xyz

    ax3 = gp_Ax3(center, gp_Dir(obb.XDirection()), gp_Dir(obb.YDirection()))
    T = gp_Trsf()
    T.SetTransformation(ax3)
    loc = TopLoc_Location(T)
    new_shape = shape.Located(loc)

    vol_prop = GProp_GProps()
    sur_prop = GProp_GProps()
    brepgprop_VolumeProperties(new_shape, vol_prop)
    brepgprop_SurfaceProperties(new_shape, sur_prop)

    matrixOfInertia = vol_prop.MatrixOfInertia()
    Ixx = matrixOfInertia.Value(1, 1)
    Iyy = matrixOfInertia.Value(2, 2)
    Izz = matrixOfInertia.Value(3, 3)
    Ixxyyzz = np.sort(np.array([Ixx, Iyy, Izz]))
    Ixy = abs(matrixOfInertia.Value(1, 2))
    Ixz = abs(matrixOfInertia.Value(1, 3))
    Iyz = abs(matrixOfInertia.Value(2, 3))
    Ixyxzyz = np.sort(np.array([Ixy, Ixz, Iyz]))
    if np.sum(Ixxyyzz) != 0:
        par[3:6] = Ixxyyzz / np.sum(Ixxyyzz)
    if np.sum(Ixyxzyz) != 0:
        par[6:9] = Ixyxzyz / np.sum(Ixyxzyz)
    par_scale[3:6] = Ixxyyzz
    par_scale[6:9] = Ixyxzyz

    face_exp = TopExp_Explorer(new_shape, TopAbs_FACE)
    edge_exp = TopExp_Explorer(new_shape, TopAbs_EDGE)
    vert_exp = TopExp_Explorer(new_shape, TopAbs_VERTEX)
    wire_exp = TopExp_Explorer(new_shape, TopAbs_WIRE)
    N = np.array([count_shape(face_exp), count_shape(edge_exp), count_shape(vert_exp), count_shape(wire_exp)], dtype=float)
    par[9:13] = N / np.sum(N)
    par_scale[9:13] = N / np.sum(N)

    vol = vol_prop.Mass()
    area = sur_prop.Mass()

    box = convert_bnd_to_shape(obb)
    vol_prop_box = GProp_GProps()
    sur_prop_box = GProp_GProps()
    brepgprop_VolumeProperties(box, vol_prop_box)
    brepgprop_SurfaceProperties(box, sur_prop_box)
    box_vol = vol_prop_box.Mass()
    box_area = sur_prop_box.Mass()

    par[13] = vol / box_vol
    par[14] = area / box_area
    par_scale[13] = vol
    par_scale[14] = area

    center_vertex = BRepBuilderAPI_MakeVertex(center).Shape()
    if face_exp.Value():
        face_prop = GProp_GProps()
        brepgprop_SurfaceProperties(face_exp.Value(), face_prop)
        par[16] = face_prop.Mass()
        par[15] = min_dist(center_vertex, face_exp.Value())
        while face_exp.More():
            face_exp.Next()

            if min_dist(center_vertex, face_exp.Value()) < par[15]:
                par[15] = min_dist(center_vertex, face_exp.Value())
            brepgprop_SurfaceProperties(face_exp.Value(), face_prop)
            if face_prop.Mass() > par[16]:
                if face_exp.More():
                    par[16] = face_prop.Mass()
        face_exp.ReInit()
        par_scale[15] = par[15]
        par_scale[16] = par[16]
        par[15] /= xyz[0]
        par[16] /= area
        classifier = BRepClass3d_SolidClassifier(shape, center, 1e-6)
        if classifier.State() != TopAbs_IN:
            par[15] = -par[15]
            par_scale[15] = -par_scale[15]

    return par, par_scale


def calc_par(data_dir, meta):
    par_list = []
    scale_par_list = []
    for d in meta:
        if os.path.exists(os.path.join(data_dir, d['category'], 'STP', d['name']+'.stp')):
            path = os.path.join(data_dir, d['category'], 'STP', d['name']+'.stp')
        elif os.path.exists(os.path.join(data_dir, d['category'], 'STEP', d['name'] + '.stp')):
            path = os.path.join(data_dir, d['category'], 'STEP', d['name'] + '.stp')
        elif os.path.exists(os.path.join(data_dir, d['category'], 'STEP', d['name'] + '.step')):
            path = os.path.join(data_dir, d['category'], 'STEP', d['name'] + '.step')
        elif os.path.exists(os.path.join(data_dir, d['category'], 'STP', d['name'] + '.step')):
            path = os.path.join(data_dir, d['category'], 'STP', d['name'] + '.step')
        else:
            d['par'] = np.zeros(17, dtype=float)
            d['scale_par'] = np.zeros(17, dtype=float)
            continue
        model = read_step_file(path)
        par, scale_par = get_par(model)
        d['par'] = par
        d['scale_par'] = scale_par
        print(d['name'])
        print(par)
        print(scale_par)
    return np.array(par_list), np.array(scale_par_list)