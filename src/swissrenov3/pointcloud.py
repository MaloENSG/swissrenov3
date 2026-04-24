# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:35:01 2026

@author: malo.delacour
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import copy
import laspy
from datetime import datetime

# # Charger le fichier LAS/LAZ
# file_path = "D:\\SWISSRENOV\\process_level\\test_shape_3.las"  # remplace par ton fichier
# las = laspy.read(file_path)

# # Afficher les dimensions disponibles
# print("=== Dimensions disponibles ===")
# for dim in las.point_format.dimensions:
#     name = dim.name
#     dtype = dim.dtype  # type numpy (ex: int32, float64, uint16...)

#     print(f"{name} : {dtype}")
    
    
class PointCloud:
    def __init__(self, xyz, rvb, classification=None, indexation=None, info=None):
        self.xyz = xyz
        self.rvb = rvb
        self.classification = classification
        self.indexation = indexation
        self.info = info or PointCloudInfo()
        
    #### check validity and data 
    
    def is_classified(self):
        """Retourne True si le nuage est classifié"""
        return self.classification is not None
    
    def is_indexed(self):
        """Retourne True si le nuage est indexé"""
        return self.indexation is not None
    
    def is_len_valid(self):
        """Vérifier si la taille des attributs sont cohérents"""
        if self.xyz is None or self.rvb is None:
            return False
        
        n = len(self.xyz)
        
        if len(self.rvb) != n:
            return False
        
        if self.is_classified() and len(self.classification) != n:
            return False
        
        if self.is_indexed() and len(self.indexation) != n:
            return False
    
        return True
    
    #### size info
    
    def __len__(self):
        return len(self.xyz)
    
    def index(self):
        """Calcul l'indexation du nuage de points"""
        self.indexation = np.arange(0, len(self.xyz))
        
    def classify(self, classif):
        """Affecte une classification au nuage."""
        
        # Tester la validité de la classification
        if not isinstance(classif, np.ndarray):
            raise TypeError("Type error : besoin d'un array")
        elif len(self._xyz) != len(classif):
            raise TypeError(f"Size error : taille classif {len(classif)}, besoin de taille {self._xyz}")
        
        # ajout de la classification
        self.classification = classif

        
        
    #### geometry info
    
    def bbox(self):
        """Retourne la boite englobante du nuage de points.
        
        Convention : X = droite, Y = profondeur, Z = hauteur
        - topLeft     : x_min, z_max  (haut gauche)
        - topRight    : x_max, z_max  (haut droite)
        - bottomLeft  : x_min, z_min  (bas gauche)
        - bottomRight : x_max, z_min  (bas droite)
        Y = y_min (face avant)
        """
        val_min = np.min(self.xyz, axis=0)  # [x_min, y_min, z_min]
        val_max = np.max(self.xyz, axis=0)  # [x_max, y_max, z_max]
    
        x_min, y_min, z_min = val_min
        x_max, y_max, z_max = val_max
    
        bbox = np.array([
            [x_min, y_max, z_max],  # topLeft
            [x_max, y_max, z_max],  # topRight
            [x_min, y_min, z_min],  # bottomLeft
            [x_max, y_min, z_min],  # bottomRight
        ])
    
        return bbox
    
    def centroid(self):
        """Centre de masse du nuage."""
        return np.mean(self.xyz, axis=0)
    
    def extent(self):
        """Dimensions (x, y, z) de la bbox."""
        return np.max(self.xyz, axis=0) - np.min(self.xyz, axis=0)
    
    #### export PointCloud
    
    def write_las(self, path):
        """
        Exporte un PointCloud en fichier .las / .laz
        
        Args:
            path        : chemin de sortie (ex: "scan.las" ou "scan.laz")
            pc          : objet PointCloud
        """
        if not self.is_len_valid():
            raise ValueError("PointCloud invalide : tailles incohérentes")

        # 1. Header
        header = laspy.LasHeader(point_format=2, version="1.2")
        las = laspy.LasData(header=header)

        # 2. XYZ
        las.x = self.xyz[:, 0]
        las.y = self.xyz[:, 1]
        las.z = self.xyz[:, 2]

        # 3. RVB 
        rvb = self.rvb.astype(np.uint16)
        las.red   = rvb[:, 0]
        las.green = rvb[:, 1]
        las.blue  = rvb[:, 2]

        # 4. Classification — si présente
        if self.is_classified():
            las.classification = self.classification
            
        # 5. Indexation
        if self.is_indexed():
            las.add_extra_dim(laspy.ExtraBytesParams(name="indexation", type=np.int32))
            las["indexation"] = self.indexation

        # 6. Ecriture
        las.write(path)
    
    
    

class Referentiel:
    """Référentiel local défini par un offset (x, y, z) et un angle de rotation (degré)."""

    def __init__(self, x=0, y=0, z=0, a=0):
        self.x = float(x)  # passe par les setters pour validation
        self.y = float(y)
        self.z = float(z)
        self.a = float(a)
        
    #### check validity and data 
    
    def is_default(self):
        """Retourne True si le référentiel est à l'origine sans rotation."""
        return self.x == 0 and self.y == 0 and self.z == 0 and self.a == 0
    
    @property
    def offset(self):
        """Retourne l'offset comme array [x, y, z]."""
        return np.array([self.x, self.y, self.z])
    


class PointCloudInfo:
    """Informations liées au nuage de points """

    def __init__(self):
        # --- Identification ---
        self.id = None
        self.name = ""
        self.date = ""
        self.crs = ""          # ex: "EPSG:2056"
        
        # --- Source ---
        self.source_sensor = ""   # SCA scanner; PHO photogrammetry; MIX mixte
        self.is_merged = False
        self.source_id = []

        # --- Caractéristiques ---
        self.sampling = None   # pas d'échantillonnage en mètres
        self.is_filtered = False
        self.is_indexed = False
        self.source_index = None

        # --- Scanner ---
        self.scan_pos = None   # np.array([x, y, z])
        self.scan_rot = None   # np.array([rx, ry, rz]) en radians

        # --- Emprise ---
        self.extent_bbox = None    # 4 ou 8 points
        self.extent_poly = None    # contour 2D
        
        # --- referentiel ---
        self.offset = [0, 0, 0]
        self.angle = 0
        
        # --- Traçabilité ---
        self._history: list[str] = []
        
        def add_history(self, message: str):
            """Ajoute une entrée dans l'historique des traitements."""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._history.append(f"[{timestamp}] {message}")
    
        @property
        def history(self):
            return self._history
        

class Raster:
    """Raster 2D généré à partir d'un PointCloud."""
    
    def __init__(self, raster: np.ndarray, resolution=None, gridsize=None,mode="", axis= "", offset=[0, 0, 0], angle=0.0):
        
        self.raster = raster
        self.resolution = resolution  # taille d'un pixel en mètres
        self.gridsize = gridsize      # (x_min, x_max, y_min, y_max)
        self.mode = mode          # 'm', 'a', 'c', 'p'
        self.axis = axis
        
        # --- referentiel du PointCloud ---
        self.offset = offset      # offset du PointCloud
        self.angle = angle        # rotation en degrés du PointCloud

    @property
    def shape(self):
        return self.raster.shape

    @property
    def width(self):
        return self.raster.shape[1]

    @property
    def height(self):
        return self.raster.shape[0]

    @property
    def is_color(self):
        """True si le raster est en couleur (mode 'c')."""
        return self.raster.ndim == 3

    def pixel_to_world(self, px: int, py: int) -> tuple[float, float]:
        """Convertit des coordonnées pixel en coordonnées monde."""
        x_min, _, _, y_max = self.gridsize
        x = x_min + px * self.resolution
        y = y_max - py * self.resolution
        return x, y

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convertit des coordonnées monde en coordonnées pixel."""
        x_min, _, _, y_max = self.gridsize
        px = int((x - x_min) / self.resolution)
        py = int((y_max - y) / self.resolution)
        return px, py









