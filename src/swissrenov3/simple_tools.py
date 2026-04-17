# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:04:05 2026

@author: malo.delacour
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import argrelmax
from .pointcloud import PointCloud, Raster
from .o3d_tools import pc_normals
from .utils import *

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
    # Gestion axe négatif -> flip avant projection
    if axis.startswith("-"):
        pc  = flip_pc(pc, axis[1])   # ex: "-x" -> flip sur "x"
        axis_abs = axis[1]               # ex: "-x" -> "x"
    else :
        axis_abs = axis

    axes = {
        "x" : (pc.xyz[:, 1], pc.xyz[:, 2], pc.xyz[:, 0]),
        "y" : (pc.xyz[:, 0], pc.xyz[:, 2], pc.xyz[:, 1]),
        "z" : (pc.xyz[:, 0], pc.xyz[:, 1], pc.xyz[:, 2]),
    }

    x, y, z = axes[axis_abs]

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
    if axis in ['-x', 'y', '-z']:
        raster = np.flip(raster,1)

    return Raster(
        raster=raster, 
        resolution=resolution, 
        gridsize=(x_min, x_max, y_min, y_max),
        mode=mode,
        axis=axis, 
        offset=pc.info.offset,
        angle=pc.info.angle
        )


def pc_raster_layer(pc: PointCloud, axis: str, resolution: float, 
                    step: float, width: float, 
                    interval: tuple = None) -> Raster:
    """
    Superpose des accumulations de tranches pour créer un raster de densité par couches.
    
    Args:
        pc       : PointCloud source
        axis     : axe de vue "x", "-x", "y", "-y", "z", "-z"
        resolution: taille d'une cellule en mètres
        step     : intervalle entre chaque tranche en mètres
        width    : largeur de chaque tranche en mètres
        interval : (val_min, val_max) en mètres, défaut = étendue du nuage
    
    Returns:
        Raster : accumulation des tranches
    """
    # Axe de découpe (index dans xyz)
    if axis.startswith("-"):
        axe = str_to_axis(axis[1])
    else :
        axe = str_to_axis(axis)

    # Intervalle de découpe
    if interval is None:
        val_min = pc.xyz[:, axe].min()
        val_max = pc.xyz[:, axe].max()
    else:
        val_min, val_max = interval

    # Grille de base
    gridsize = make_grid(pc, axe)
    x_min, x_max, y_min, y_max = gridsize
    grid_width  = int(np.ceil((x_max - x_min) / resolution))
    grid_height = int(np.ceil((y_max - y_min) / resolution))
    raster_acc  = np.zeros((grid_height, grid_width))

    # Itération sur les tranches
    steps = np.arange(val_min, val_max, step)
    for i, val in enumerate(steps):
        print(f"Tranche {i+1}/{len(steps)} : {val:.3f} m")

        # Crop de la tranche
        idx_layer, _ = select_crop(pc, **{
            f"{['x','y','z'][axe]}_peak" : val + width / 2,
            f"{['x','y','z'][axe]}_width": width,
        })

        if len(idx_layer) == 0:
            continue

        pc_layer = select_pc_index(pc, idx_layer)

        # Rasterisation de la tranche
        axis_abs = axis_to_str(axe)
        r = pc_rasterise(pc_layer, mode='p', resolution=resolution,
                         axis=axis_abs, grid_size=gridsize)
        raster_acc += r.raster
        
    # Flip final
    raster_acc = raster_acc
    
    if axis.startswith("-"):
        raster_acc = np.flip(raster_acc,1)

    return Raster(
        raster     = raster_acc,
        resolution = resolution,
        gridsize   = gridsize,
        mode       = 'layer',
        axis       = axis,
        offset     = pc.info.offset,
        angle      = pc.info.angle,
    )

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


def raster_to_image(raster: Raster, path: str, 
                    colormap: str = "gray") -> np.ndarray:
    """
    Convertir un Raster en image avec colormap.
    
    Args:
        raster   : objet Raster source
        colormap : "binary" | "gray" | "afmhot" | "bwr" | "jet"
    
    Returns:
        img : np.ndarray image exportée (BGR pour cv2)
    """
    COLORMAPS = ["binary", "gray", "afmhot", "bwr", "jet"]
    if colormap not in COLORMAPS:
        raise ValueError(f"Colormap invalide : '{colormap}', valeurs acceptées : {COLORMAPS}")

    data = raster.raster

    # 1. Normalisation [0, 1]
    if raster.is_color:
        # Mode couleur 'c' : déjà en [0, 1], pas de colormap
        img = (data * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        # Normalisation
        r_min, r_max = data.min(), data.max()
        if r_max > r_min:
            data_norm = (data - r_min) / (r_max - r_min)
        else:
            data_norm = np.zeros_like(data)

        # 2. Application colormap matplotlib
        cmap = plt.get_cmap(colormap)
        img_rgba = (cmap(data_norm) * 255).astype(np.uint8)  # (H, W, 4) RGBA
        img_rgb  = img_rgba[:, :, :3]                         # (H, W, 3) RGB
        img      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)   # BGR pour cv2

    # 4. World file
    # x_min, _, _, y_max = raster.gridsize
    # tfw_path = os.path.splitext(path)[0] + tfw_extension(path)
    # write_tfw(tfw_path, raster.resolution, x_min, y_max, raster.angle)

    # print(f"Exporté : {path} + {tfw_path}")
    return img

def export_image(img, path):
    """
    Convertir un Raster en image avec colormap.
    
    Args:
        img  : np.ndarray image exportée (BGR pour cv2)
        path : chemin de sortie avec extension (.png, .jpg, .tif)
    """
    ext = os.path.splitext(path)[1].lower()
    FORMATS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    if ext not in FORMATS:
        raise ValueError(f"Format invalide : '{ext}', valeurs acceptées : {FORMATS}")
    
    cv2.imwrite(path, img)
    return True


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














