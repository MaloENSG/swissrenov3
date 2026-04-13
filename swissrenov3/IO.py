# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:39:05 2026

@author: malo.delacour
"""

import laspy
# import pye57
import numpy as np
from pointcloud import PointCloud, PointCloudInfo, Referentiel

def read_las(path):
    """Charge un fichier .las / .laz et retourne un PointCloud."""
    
    las = laspy.read(path)
    
    # XYZ
    xyz = np.vstack([las.x, las.y, las.z]).T
    
    # RVB — normalisation 8bits
    if np.any(las.red != 0):
        rvb = np.vstack([las.red, las.green, las.blue]).T
        rvb = (rvb / rvb.max() * 255).astype(np.uint8)
    else:
        rvb = np.zeros((len(xyz), 3), dtype=np.uint8)
    
    # Classification — pas toujours présente
    classification = None
    if np.any(las.classification != 0):
        classification = np.array(las.classification)
        
    # Indexation — pas toujours présente
    indexation = None
    if hasattr(las, 'indexation'):
        indexation = np.array(las.classification)
    
    return PointCloud(xyz, rvb, classification=classification, indexation=indexation)


    











