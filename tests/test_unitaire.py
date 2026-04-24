# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:56:43 2026

@author: malo.delacour
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import swissrenov3 as s3
import open3d as o3d

rt = "D:/SWISSRENOV/AutoProcessing/autoswiss/"

ipath = rt + "exemple_scan.las"
ipath = "D:/SWISSRENOV/samples_tests/gigon/gigon_10mm.las"
pc = s3.IO.read_las(ipath)

# pc.index()

# # Création d'un référentiel
# ref = s3.pointcloud.Referentiel(x=2588822.217, 
#                   y=1242639.783, 
#                   z=451.861, 
#                   a=1.32)

# pc_loc, bbox_glob = s3.geometry.refGlob2refLoc(pc, ref)

# idx_class, idx_out = u.select_by_class(pc_loc, [1, 2])
# pc_class = u.select_pc_index(pc_loc, idx_class)


# ras = s3.simple_tools.pc_rasterise(pc_loc, mode='a', resolution=0.01, axis='x', grid_size=None)

# ras = s3.simple_tools.pc_raster_layer(pc_loc, axis="y", resolution=0.03, 
#                     step=0.7, width=0.3)

# plt.imshow(ras.raster)

# img = s3.simple_tools.raster_to_image(ras, "scan.png",  colormap="jet")


# opath = "exout.las"
# pc_class.write_las(opath)

# start = time.time()
# pcd = s3.o3d_tools.rg_cluster(pc, tangle=10)
# end = time.time()

# print(end-start)

# o3d.visualization.draw_geometries([pcd])



# unique_labels, inverse = np.unique(idx_labels, return_inverse=True)
# k = unique_labels.shape[0]

# lut = np.random.randint(0, 256, size=(k, 3), dtype=np.uint8)
# colors = lut[inverse]

# pc.indexation = idx_labels
# opath = "test.las"
# pc.write_las(opath)




