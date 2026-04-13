# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:04:05 2026

@author: malo.delacour
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from pointcloud import PointCloud, Raster
from o3d_tools import pc_normals
import utils

import os

def pc_main_orientation(pc: PointCloud, bin_width=0.05, plot=False):
    """
    Détecte l'angle principal d'orientation en Z d'un PointCloud.
    
    Args:
        pc        : PointCloud source
        bin_width : résolution de l'histogramme en degrés
        plot      : affiche l'histogramme si True
    
    Returns:
        angle : angle principal en degrés [0, 90]
    """
    
    nn=16
    radius=0.24
    
    # 1. Calcul des normales
    normals = pc_normals(pc, nn=nn, radius=radius)

    # 2. Orientations en Z -> [0, 90]
    ori = np.arctan2(normals[:, 1], normals[:, 0])  # arctan2 plus robuste que arctan
    ori = np.rad2deg(ori) % 90

    # 3. Histogramme
    bins = np.arange(ori.min(), ori.max() + bin_width, bin_width)
    hist, angles = np.histogram(ori, bins=bins)

    # 4. Détection des modes
    max_inds = argrelmax(hist, order=12)[0]
    nb_modes = max_inds.size

    if nb_modes == 0:
        # Aucun mode détecté -> pic global
        angle = angles[np.argmax(hist)]
    elif nb_modes == 1:
        angle = angles[max_inds[0]]
    else:
        # Plusieurs modes -> le plus dominant
        angle = angles[max_inds[np.argmax(hist[max_inds])]]

    print(f"Nb modes détectés : {nb_modes} | Angle principal : {angle:.2f}°")

    # 5. Affichage optionnel
    if plot:
        plt.figure(figsize=(8, 3))
        plt.bar(angles[:-1], hist, width=bin_width, align='edge', alpha=0.7, color='blue')
        plt.axvline(angle, color='red', linewidth=2, label=f"Angle : {angle:.2f}°")
        plt.xlabel("Orientation (°)")
        plt.ylabel("Nombre de points")
        plt.title("Histogramme des orientations")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return angle

def pc_flip(pc: PointCloud, axis):
    """
    Adapter le PointCloud suivant l'axe'
    
    Args:
        pc         : PointCloud source
        axis       : axe de projection "x", "-x", "y", "-y", "z", "-z"
    
    Returns:
        pc_flipped : PointCloud source
    """
    
    axes = {
        "x" : ( pc.xyz[:, 1],  pc.xyz[:, 2],  pc.xyz[:, 0]),
        "-x": ( pc.xyz[:, 1],  pc.xyz[:, 2], -pc.xyz[:, 0]),
        "y" : ( pc.xyz[:, 0],  pc.xyz[:, 2],  pc.xyz[:, 1]),
        "-y": ( pc.xyz[:, 0],  pc.xyz[:, 2], -pc.xyz[:, 1]),
        "z" : ( pc.xyz[:, 0],  pc.xyz[:, 1],  pc.xyz[:, 2]),
        "-z": ( pc.xyz[:, 0], -pc.xyz[:, 1],  pc.xyz[:, 2]),
    }
    
    x, y, z = axes[axis]
    xyz = np.hstack(x, y, z)
    
    return PointCloud(
        xyz            = xyz,
        rvb            = pc.rvb,
        classification = pc.classification if pc.is_classified() else None,
        indexation     = pc.indexation     if pc.is_indexed()    else None,
    )

def pc_rasterise(pc: PointCloud, mode, resolution, axis, grid_size=None):
    """
    Conversion PointCloud -> Raster. Rasterise un PointCloud en image 2D.
    
    Args:
        pc         : PointCloud source
        mode       : 'm' = hauteur max | 'a' = accumulation   | 
                     'c' = couleurs    | 'p' = masque binaire | 
                     'n' = hauteur min | 'e' = rugosité
        resolution : taille d'une cellule en mètres
        axis       : axe de projection "x", "-x", "y", "-y", "z", "-z"
        grid_size  : (x_min, x_max, y_min, y_max) optionnel
    
    Returns:
        raster : Raster
    """
    if mode == '' or len(mode) > 1:
        raise ValueError("Mode vide : choisir parmi 'm', 'a', 'c', 'p', 'n', 'e' ")

    # 1. Sélection des axes et gestion du signe
    axes = {
        "x" : ( pc.xyz[:, 1],  pc.xyz[:, 2],  pc.xyz[:, 0]),
        "-x": ( pc.xyz[:, 1],  pc.xyz[:, 2], -pc.xyz[:, 0]),
        "y" : ( pc.xyz[:, 0],  pc.xyz[:, 2],  pc.xyz[:, 1]),
        "-y": ( pc.xyz[:, 0],  pc.xyz[:, 2], -pc.xyz[:, 1]),
        "z" : ( pc.xyz[:, 0],  pc.xyz[:, 1],  pc.xyz[:, 2]),
        "-z": ( pc.xyz[:, 0], -pc.xyz[:, 1],  pc.xyz[:, 2]),
    }
    
    if axis not in axes:
        raise ValueError(f"Axe invalide : '{axis}', valeurs acceptées : {list(axes.keys())}")

    x, y, z = axes[axis]

    # 2. Normalisation de z (toujours positif, part de 0)
    z = z - z.min()

    # 3. Grille
    if grid_size is None:
        x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    else:
        x_min, x_max, y_min, y_max = grid_size

    grid_width  = int(np.ceil((x_max - x_min) / resolution))
    grid_height = int(np.ceil((y_max - y_min) / resolution))

    # 4. Indices grille
    x_idx = np.clip(((x - x_min) / resolution).astype(int), 0, grid_width  - 1)
    y_idx = np.clip(((y - y_min) / resolution).astype(int), 0, grid_height - 1)

    # 5. Remplissage selon mode
    if 'c' in mode:
        raster_z = np.full((grid_height, grid_width), -np.inf)
        raster   = np.zeros((grid_height, grid_width, 3))
        rvb      = pc.rvb / 255.0 if pc.rvb.max() > 1 else pc.rvb
        for xi, yi, zi, color in zip(x_idx, y_idx, z, rvb):
            if zi > raster_z[yi, xi]:
                raster_z[yi, xi] = zi
                raster[yi, xi]   = color
    else:
        raster = np.zeros((grid_height, grid_width))
        if 'm' in mode:
            for xi, yi, zi in zip(x_idx, y_idx, z):
                raster[yi, xi] = max(raster[yi, xi], zi)
        if 'a' in mode:
            np.add.at(raster, (y_idx, x_idx), 1)   # vectorisé, plus rapide que le for
        if 'p' in mode:
            raster[y_idx, x_idx] = 1                # vectorisé
        if 'n' in mode:
            raster = np.full((grid_height, grid_width), np.inf)
            for xi, yi, zi in zip(x_idx, y_idx, z):
                raster[yi, xi] = min(raster[yi, xi], zi)
            raster[raster == np.inf] = 0
        if 'e' in mode:
            count  = np.zeros((grid_height, grid_width))
            sum_z  = np.zeros((grid_height, grid_width))
            sum_z2 = np.zeros((grid_height, grid_width))
            np.add.at(count,  (y_idx, x_idx), 1)
            np.add.at(sum_z,  (y_idx, x_idx), z)
            np.add.at(sum_z2, (y_idx, x_idx), z**2)
            mean = np.divide(sum_z, count, where=count > 0)
            raster = np.sqrt(np.divide(sum_z2, count, where=count > 0) - mean**2)

    # 6. Flip pour orientation image naturelle
    raster = raster[::-1]

    return Raster(
        raster=raster, 
        resolution=resolution, 
        gridsize=(x_min, x_max, y_min, y_max),
        mode=mode,
        axis=axis, 
        offset=pc.info.offset,
        angle=pc.info.angle
        )


def pc_raster_layer(pcd_pcd, axis_view, resolution, pas_val, largeur_layer, inter_val=None):
    """
    

    Parameters
    ----------
    pcd_pcd : TYPE
        Nuage de point.
    axis_view : TYPE
        axe de vue.
    resolution :
        resolution du raster
    pas_val : TYPE
        interval entre chaque tranche.
    largeur_layer : TYPE
        largeur de tranche.
    inter_val : TYPE, optional
        Interval entre la premiere et derniere tranche (en cm).
        par defaut, du min au max du nuage de points. The default is None.

    Returns
    -------
    raster : TYPE
        DESCRIPTION.

    """
    
    round_value = 100 # cm -> 1000 pour le mm

    ####
    if axis_view==0 or axis_view==-1:
        axis = 0
    elif axis_view==1 or axis_view==-2:
        axis = 1
    elif axis_view==2 or axis_view==-3:
        axis = 2
    
    np_pcd = np.asarray(pcd_pcd.points)
    raster, gridsize = make_grid(pcd_pcd, resolution=resolution, axis=axis)
    
    if inter_val == None:
        val_min, val_max = int(np.min(np_pcd, axis=0)[axis]*round_value), int(np.max(np_pcd, axis=0)[axis]*round_value)
    else :
        val_min, val_max = inter_val
    
    
    pas_val = int(pas_val*round_value)
    for val in range(val_min, val_max, pas_val):
        
        valm = val/round_value
        print(val)
        
        pcd_layer, invert_layer_pcd = extract_layer(pcd_pcd, valm, largeur_layer, axis=axis, k=0.75)
        
        img = rasterise_xyz(pcd_layer, mode='p', resolution=resolution, axis=axis_view, grid_size=gridsize)
        raster += img
        
    return raster



def write_tfw(raster: Raster, path: str):
    """
    Ecrit un fichier .tfw de géoréférencement.
    
    Args:
        path       : chemin du fichier .tfw
        resolution : taille d'un pixel en mètres
        x_min      : coordonnée X du coin supérieur gauche
        y_max      : coordonnée Y du coin supérieur gauche
        angle      : angle de rotation en degrés (défaut 0)
    
    Format TFW :
        A : taille pixel X * cos(angle)
        D : taille pixel X * sin(angle)
        B : taille pixel Y * sin(angle)
        E : taille pixel Y * -cos(angle)  (négatif)
        C : X coin supérieur gauche
        F : Y coin supérieur gauche
    """
    gridsize = raster.gridsize
    resolution = raster.resolution
    angle = raster.angle
    
    x_min, x_max, y_min, y_max = gridsize
    theta = np.deg2rad(angle)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    A = resolution * cos_a
    D = resolution * sin_a
    B = resolution * sin_a
    E = -resolution * cos_a
    C = x_min
    F = y_max

    with open(path, 'w') as f:
        for value in [A, D, B, E, C, F]:
            f.write(f"{value:.6f}\n")


def tfw_extension(image_path: str):
    """
    Trouver l'extention de géoréférencementdu fichier world à partir de l'extention d'image
    Format world :
        png  : .pgw
        jpg  : .jgw
        jpeg : .jgw
        tif  : .tfw
        tiff : .tfw
    """
    
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".png":  ".pgw",
        ".jpg":  ".jgw",
        ".jpeg": ".jgw",
        ".tif":  ".tfw",
        ".tiff": ".tfw",
    }.get(ext, ".txt")

# def export_raster(path, raster, )














