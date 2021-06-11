import numpy as np
import cv2
from icecream import ic

def export(points, positions, frame):
    header = """ply
        format ascii 1.0
        element vertex {0}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    """
    # ic(cv2.minMaxLoc(dip))
    pcl = []
    # ic(points.shape)
    # ic(positions.shape)
    # for index, (point, position) in enumerate(zip(points, positions)):
        # ic(point)
        # ic(position)
    for uy, column in enumerate(points):
        for ux, pos3d in enumerate(column):
            b, g, r = frame[uy, ux]
            px,py,pz = pos3d
            x, y, z = ux, uy, np.linalg.norm(pos3d)
            if np.isfinite(px) and np.isfinite(py) and np.isfinite(pz) and z<500:
                # pcl.append(f"{px} {py} {pz} {int(abs(px)/100)} {int(abs(py)/100)} {int(abs(pz)/100)}")
                pcl.append(f"{px} {py} {pz} {r} {g} {b}")
                # pcl.append(f"{x+.1} {y+.1} {z} {r} {g} {b}")

    ply = header.format(len(pcl)) + "\n".join(pcl)
    with open("test.ply", "w") as fp:
        fp.write(ply)
    print("done")