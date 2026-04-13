# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:05:49 2026

@author: malo.delacour
"""

import numpy as np
import open3d as o3d
from pointcloud import PointCloud, PointCloudInfo, Referentiel

def to_open3d(pc: PointCloud):
    """Convertit un PointCloud en objet Open3D."""
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.xyz.astype(np.float64)) # XYZ
    
    # RVB — normalisation
    rvb = pc.rvb.astype(np.float64)
    if rvb.max() > 1.0:
        rvb = rvb / 255.0
    pcd.colors = o3d.utility.Vector3dVector(rvb)

    return pcd

def from_open3d(pcd: o3d.geometry.PointCloud):
    """Convertit un objet Open3D en PointCloud."""
    
    xyz = np.asarray(pcd.points).astype(np.float32)
    rvb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    
    return PointCloud(xyz, rvb)

def select_outliers(pc: PointCloud, nn=6, std_multiplier=2):
    """
    Retourne les indices des points valides et des outliers.
    
    Args:
        pc             : PointCloud source
        nn             : nombre de voisins pour le calcul statistique
        std_multiplier : seuil en écart-type (plus bas = plus agressif)
    
    Returns:
        idx_clean    : indices des points conservés
        idx_outliers : indices des points aberrants
    """
    
    pcd = to_open3d(pc)
    
    # Filtrage
    _, idx_clean = pcd.remove_statistical_outlier(nn, std_multiplier)
    idx_clean = np.array(idx_clean)
    
    all_idx      = np.arange(len(pc))
    idx_outliers = np.setdiff1d(all_idx, idx_clean)
    
    return idx_clean, idx_outliers


def pc_normals(pc: PointCloud, nn=16, radius=0.24):
    """
    Calcule les normales d'un nuage de points.
    
    Args:
        pc     : PointCloud source
        nn     : nombre de voisins maximum
        radius : rayon de recherche en mètres
    
    Returns:
        normals : array (n, 3) des vecteurs normaux
    """
    
    pcd = to_open3d(pc)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn), fast_normal_computation=True)
    normals = np.asarray(pcd.normals)
    
    return normals















