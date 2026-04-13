# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:56:43 2026

@author: malo.delacour
"""

import numpy as np
import matplotlib.pyplot as plt
from pointcloud import PointCloud, PointCloudInfo, Referentiel, Raster
import IO as io
import geometry as geo
import utils as u

import simple_tools as st

ipath = "exemple.las"
pc = io.read_las(ipath)

pc.index()

# Création d'un référentiel
ref = Referentiel(x=2588822.217, 
                  y=1242639.783, 
                  z=451.861, 
                  a=1.32)

pc_loc, bbox_glob = geo.refGlob2refLoc(pc, ref)

idx_class, idx_out = u.select_by_class(pc_loc, [1, 2])
pc_class = u.select_pc_index(pc_loc, idx_class)


ras = st.pc_rasterise(pc_loc, mode='c', resolution=0.03, axis='x', grid_size=None)

plt.imshow(ras.raster)


# opath = "exout.las"
# pc_class.write_las(opath)


