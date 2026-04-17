# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:05:49 2026

@author: malo.delacour
"""

import numpy as np
import open3d as o3d
from .pointcloud import PointCloud, PointCloudInfo, Referentiel
from .geometry import pc_translate

import matplotlib.pyplot as plt

from collections import deque

def to_open3d(pc: PointCloud):
    """Convertit un PointCloud en objet Open3D."""
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.xyz.astype(np.float32)) # XYZ
    
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



############
# Code issu de https://medium.com/point-cloud-python-matlab-cplus/area-growing-clustering-algorithm-for-point-clouds-with-open3d-python-code-0358137a40b5
 
def angle2p(N1, N2):
    # Input two normals, return the angle
    dt = N1[0] * N2[0] + N1[1] * N2[1] + N1[2] * N2[2]
    dt = np.arccos(np.clip(dt, -1, 1))
    r_Angle = np.degrees(dt)
    return r_Angle


class RegionGrowing:
    def __init__(self):
        """
           Init parameters
        """
        self.pcd = None  # input point clouds
        self.NPt = 0  # input point clouds
        self.nKnn = 20  # normal estimation using k-neighbour
        self.nRnn = 0.1  # normal estimation using r-neighbour
        self.rKnn = 20  # region growing using k-neighbour
        self.rRnn = 0.1  # region growing using r-neighbour
        self.pcd_tree = None  # build kdtree
        self.TAngle = 5.0
        self.Clusters = []
        self.minCluster = 100  # minimal cluster size
 
    def SetDataThresholds(self, pc, t_a=10.0):
        self.pcd = pc
        self.TAngle = t_a
        self.NPt = len(self.pcd.points)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
 
    def RGKnn(self):
        if len(self.pcd.normals) < self.NPt:
            self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.nRnn, max_nn=self.nKnn))
    
        all_normals = np.asarray(self.pcd.normals)
        all_points  = np.asarray(self.pcd.points)
        processed   = np.zeros(self.NPt, dtype=bool)
    
        for i in range(self.NPt):
            if processed[i]:
                continue
    
            seed_queue   = deque([i])
            cluster_pts  = [i]          # ← liste séparée pour accumuler
            processed[i] = True
    
            while seed_queue:
                cur        = seed_queue.popleft()
                thisNormal = all_normals[cur]
                [k, idx, _] = self.pcd_tree.search_knn_vector_3d(all_points[cur], self.rKnn)
                idx = np.asarray(idx[1:k])
    
                unprocessed = idx[~processed[idx]]
                if len(unprocessed) == 0:
                    continue
    
                dots   = np.clip(all_normals[unprocessed] @ thisNormal, -1.0, 1.0)
                angles = np.degrees(np.arccos(dots))
    
                to_add = unprocessed[angles < self.TAngle]
                processed[to_add] = True
                seed_queue.extend(to_add.tolist())
                cluster_pts.extend(to_add.tolist())  # ← on accumule ici
    
            if len(cluster_pts) > self.minCluster:   # ← on teste cluster_pts
                self.Clusters.append(cluster_pts)
 
    def ReLabeles(self):
        labels = np.zeros(self.NPt, dtype=np.int32)
        for i, cluster in enumerate(self.Clusters):
            labels[np.array(cluster)] = i + 1  # fancy indexing, pas de boucle interne
        return labels

def rg_cluster(pc: PointCloud, tangle=10):
    
    
    pc = pc_translate(pc, np.array([2593380, 1245500, 410]))
    pcd = to_open3d(pc)
    RGKNN = RegionGrowing()
    RGKNN.SetDataThresholds(pcd,tangle)
    RGKNN.RGKnn()
    labels = RGKNN.ReLabeles()
    
    max_label = len(RGKNN.Clusters)
    print(f"point cloud has {max_label} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 1] = 1         # set to white for small clusters (label - 0 )
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    return pcd
    
    











