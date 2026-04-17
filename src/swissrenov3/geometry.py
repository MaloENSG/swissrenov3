# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:55:28 2026

@author: malo.delacour
"""

import numpy as np
from .pointcloud import PointCloud, PointCloudInfo, Referentiel

def arr_translate(array: np.ndarray, offset: np.ndarray):
    """
    Applique une translation à un array (n, 3).
    
    Args:
        array  : np.ndarray (n, 3)
        offset : np.ndarray (3,)
    """
    if not isinstance(offset, np.ndarray):
        raise TypeError(f"Type error : besoin d'un np.ndarray, reçu {type(offset)}")
    if offset.shape != (3,):
        raise ValueError(f"Size error : shape {offset.shape} incorrect, besoin de (3,)")
    
    return array + offset


def arr_zrotation(array: np.ndarray, angle: float):
    """
    Applique une rotation en Z à un array (n, 3).
    
    Args:
        array : np.ndarray (n, 3)
        angle : angle en degrés (sens horaire ?)
    """
    theta = -np.deg2rad(angle)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    return array @ R_z

def pc_translate(pc: PointCloud, offset: np.ndarray) -> PointCloud:
    """Retourne un PointCloud avec une translation."""
    return PointCloud(
        xyz            = arr_translate(pc.xyz, offset),
        rvb            = pc.rvb,
        classification = pc.classification,
        indexation     = pc.indexation,
    )

def pc_zrotation(pc: PointCloud, angle: float) -> PointCloud:
    """Retourne un PointCloud avec une rotation en Z (degrés, sens horaire ?)."""
    return PointCloud(
        xyz            = arr_zrotation(pc.xyz, angle),
        rvb            = pc.rvb,
        classification = pc.classification,
        indexation     = pc.indexation,
    )

def refLoc2refGlob(pc: PointCloud):
    """
    Transforme un PointCloud du référentiel local vers le référentiel global
    à partir des données du référentiel dans PointCloudInfo.
    
    Ordre inverse : Rotation (+angle) puis Translation (+offset)
    """
    
    offset = np.array(pc.info.offset)
    angle = pc.info.angle
    
    pc_glob = pc_zrotation(pc, angle)
    pc_glob = pc_translate(pc_glob, offset)
    
    return pc_glob

def refGlob2refLoc(pc: PointCloud, ref: Referentiel):
    """
    Transforme un PointCloud du référentiel global vers le référentiel local.
    
    Ordre : Translation (-offset) puis Rotation (-angle)
    
    Returns:
        pc_loc   : PointCloud dans le référentiel local
        bbox_glob: Boite englobante locale ramenée dans le référentiel global
    """

    # Translation puis rotation du PointCloud
    pc_loc = pc_translate(pc, -ref.offset)
    pc_loc = pc_zrotation(pc_loc, -ref.a)
    
    # Sauvegarde des données de référentiel dans PoitCloudInfo
    pc_loc.info.offset = ref.offset.tolist()
    pc_loc.info.angle = ref.a
    
    # Bbox ramenée dans le référentiel global (opération inverse)
    bbox_loc = pc_loc.bbox()  
    pc_bbox = PointCloud(
        xyz = bbox_loc,
        rvb = np.zeros((len(bbox_loc), 3), dtype=np.uint8)
    )
    pc_bbox.info.offset = ref.offset.tolist()
    pc_bbox.info.angle = ref.a
    pc_bbox_glob = refLoc2refGlob(pc_bbox)
    
    return pc_loc, pc_bbox_glob.xyz 
















