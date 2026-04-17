# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:46:42 2026

@author: malo.delacour
"""

import numpy as np
from .pointcloud import PointCloud, PointCloudInfo, Referentiel

def axis_to_str(axis) -> str:
    """
    Convertit un indice d'axe en string.
    
    0  -> "x"  |  -1 -> "-x"
    1  -> "y"  |  -2 -> "-y"
    2  -> "z"  |  -3 -> "-z"
    """
    mapping = {
        0: "x",  -1: "-x",
        1: "y",  -2: "-y",
        2: "z",  -3: "-z",
    }
    if axis not in mapping:
        raise ValueError(f"Axe invalide : {axis}, valeurs acceptées : {list(mapping.keys())}")
    
    return mapping[axis]


def str_to_axis(axis: str) -> int:
    """
    Convertit un string d'axe en indice.
    
    "x"  -> 0  |  "-x" -> -1
    "y"  -> 1  |  "-y" -> -2
    "z"  -> 2  |  "-z" -> -3
    """
    mapping = {
        "x":  0,  "-x": -1,
        "y":  1,  "-y": -2,
        "z":  2,  "-z": -3,
    }
    if axis not in mapping:
        raise ValueError(f"Axe invalide : '{axis}', valeurs acceptées : {list(mapping.keys())}")
    
    return mapping[axis]

def make_grid(pc: PointCloud, axis="z"):
    """
    Calcule gridsize d'un PointCloud pour la création de grille 2D géoréférencé.
    
    Args:
        pc         : PointCloud source
        axis       : axe de projection "x", "-x", "y", "-y", "z", "-z"
    
    Returns:
        gridsize : (x_min, x_max, y_min, y_max)
        
    """
    
    if isinstance(axis, int):
        axis = axis_to_str(axis)
    
    axes = {
        "x"  : ( pc.xyz[:, 1],  pc.xyz[:, 2]),
        "-x" : (-pc.xyz[:, 1],  pc.xyz[:, 2]),
        "y"  : ( pc.xyz[:, 0],  pc.xyz[:, 2]),
        "-y" : (-pc.xyz[:, 0],  pc.xyz[:, 2]),
        "z"  : ( pc.xyz[:, 0],  pc.xyz[:, 1]),
        "-z" : ( pc.xyz[:, 0], -pc.xyz[:, 1]),
    }
    
    if axis not in axes:
        raise ValueError(f"Axe invalide : '{axis}', valeurs acceptées : {list(axes.keys())}")
    
    x, y = axes[axis]

    return x.min(), x.max(), y.min(), y.max()


def peak_width_to_range(peak, width):
    """
    Convertit un centre et une largeur en valeurs min/max.
    
    Args:
        peak  : valeur centrale — float ou np.ndarray (n,)
        width : largeur de la plage — float ou np.ndarray (n,)
    
    Returns:
        val_min, val_max
    
    Exemple:
        peak_width_to_range(10, 4) -> (8, 12)
    """
    half = width / 2
    return peak - half, peak + half


def range_to_peak_width(val_min, val_max):
    """
    Convertit des valeurs min/max en centre et largeur.
    
    Args:
        val_min : valeur minimale — float ou np.ndarray (n,)
        val_max : valeur maximale — float ou np.ndarray (n,)
    
    Returns:
        peak, width
    
    Exemple:
        range_to_peak_width(8, 12) -> (10, 4)
    """
    peak  = (val_min + val_max) / 2
    width = val_max - val_min
    return peak, width

def select_crop(pc: PointCloud,
            x_min=None, x_max=None,
            y_min=None, y_max=None,
            z_min=None, z_max=None,
            x_peak=None, x_width=None,
            y_peak=None, y_width=None,
            z_peak=None, z_width=None):
    """
    Crop un PointCloud selon une boite englobante sur X, Y, Z.
    Accepte min/max ou peak/width (convertis automatiquement).
    
    Args:
        pc              : PointCloud source
        x_min, x_max   : plage sur X
        y_min, y_max   : plage sur Y
        z_min, z_max   : plage sur Z
        x_peak, x_width: centre + largeur sur X (converti en min/max)
        y_peak, y_width: centre + largeur sur Y
        z_peak, z_width: centre + largeur sur Z
    
    Returns:
        idx      : indices des points conservés
        idx_out  : indices des points non conservés
    
    Exemple:
        select_crop(pc, x_min=0, x_max=10, z_peak=1.5, z_width=0.5)
    """
    # Conversion peak/width -> min/max si nécessaire
    if x_peak is not None and x_width is not None:
        x_min, x_max = peak_width_to_range(x_peak, x_width)
    if y_peak is not None and y_width is not None:
        y_min, y_max = peak_width_to_range(y_peak, y_width)
    if z_peak is not None and z_width is not None:
        z_min, z_max = peak_width_to_range(z_peak, z_width)

    # Masque — None = pas de contrainte sur cet axe
    mask = np.ones(len(pc), dtype=bool)

    if x_min is not None: mask &= pc.xyz[:, 0] >= x_min
    if x_max is not None: mask &= pc.xyz[:, 0] <= x_max
    if y_min is not None: mask &= pc.xyz[:, 1] >= y_min
    if y_max is not None: mask &= pc.xyz[:, 1] <= y_max
    if z_min is not None: mask &= pc.xyz[:, 2] >= z_min
    if z_max is not None: mask &= pc.xyz[:, 2] <= z_max

    idx = np.where(mask)[0]
    
    all_idx      = np.arange(len(pc))
    idx_out = np.setdiff1d(all_idx, idx)

    return idx, idx_out

def select_pc_index(pc: PointCloud, idx: np.ndarray):
    """
    Retourne un nouveau PointCloud sélectionné par indices.
    
    Args:
        pc  : PointCloud source
        idx : indices à conserver
    """
    return PointCloud(
        xyz            = pc.xyz[idx],
        rvb            = pc.rvb[idx],
        classification = pc.classification[idx] if pc.is_classified() else None,
        indexation     = pc.indexation[idx]     if pc.is_indexed()    else None,
    )

CLASSES = {
    0: "unlabeled",
    1: "ceil",
    2: "ground",
    3: "wall",
    4: "beam",
    5: "columne",
    6: "window",
    7: "door",
    8: "stairs",
    9: "clutter",
}

def class_id_to_name(class_id):
    """
    Convertit un indice de classe en nom.
    Accepte un entier ou un array numpy.
    
    Args:
        class_id : int ou np.ndarray (n,)
    
    Returns:
        str si int, list si array
    
    Exemple:
        class_id_to_name(np.array([0, 1, 2]))      -> ["non classifié", "plafond", "sol"]
    """
    if isinstance(class_id, np.ndarray):
        return [CLASSES.get(i, f"inconnu ({i})") for i in class_id]
    
    return CLASSES.get(class_id, f"inconnu ({class_id})")


def class_name_to_id(name: str):
    """
    Convertit un nom de classe en indice.
    
    Exemple:
        class_name_to_id("sol") -> 2
    """
    reverse = {v: k for k, v in CLASSES.items()}
    
    if name not in reverse:
        raise ValueError(f"Classe inconnue : '{name}', valeurs acceptées : {list(reverse.keys())}")
    
    return reverse[name]


def select_by_class(pc: PointCloud, classes):
    """
    Retourne les indices des points appartenant à une ou plusieurs classes.
    
    Args:
        pc      : PointCloud source (doit être classifié)
        classes : int ou str ou list — classe(s) à sélectionner
    
    Returns:
        idx : indices des points des classes demandées
    
    Exemple:
        crop_by_class(pc, "sol")
        crop_by_class(pc, 2)
        crop_by_class(pc, ["sol", "mur"])
        crop_by_class(pc, [2, 3])
    """
    if not pc.is_classified():
        raise ValueError("PointCloud non classifié")

    # Normalisation en liste d'indices entiers
    if not isinstance(classes, list):
        classes = [classes]
    
    
    # Convertir nom de classe en indices si besoin
    class_ids = [class_name_to_id(c) if isinstance(c, str) else c for c in classes]

    # Vérification des ids
    unknown = [c for c in class_ids if c not in CLASSES]
    if unknown:
        raise ValueError(f"Classes inconnues : {unknown}, valeurs acceptées : {list(CLASSES.keys())}")

    # Sélection
    mask    = np.isin(pc.classification, class_ids)
    idx     = np.where(mask)[0]
    idx_out = np.where(~mask)[0]

    return idx, idx_out


def flip_pc(pc: PointCloud, axis: str) -> PointCloud:
    """
    Retourne un PointCloud miroir selon un axe.
    
    Args:
        axis : "x", "y" ou "z"
    
    Returns:
        PointCloud avec l'axe inversé
    """
    flip = {"x": [-1,  1,  1],
            "y": [ 1, -1,  1],
            "z": [ 1,  1, -1]}
    
    if axis not in flip:
        raise ValueError(f"Axe invalide : '{axis}', valeurs acceptées : {list(flip.keys())}")
    
    return PointCloud(
        xyz            = pc.xyz * np.array(flip[axis]),
        rvb            = pc.rvb,
        classification = pc.classification,
        indexation     = pc.indexation,
    )















