import numpy as np
from numba import njit, prange
from scipy.interpolate import griddata

from .pointcloud import PointCloud


def readOPKFile(filename):
    """
    # Exported from Agisoft format OPK
    # Cameras (105)
    # PhotoID, X, Y, Z, Omega, Phi, Kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33
    """
    
    out = {}
    with open(filename) as f:
        for l in f:
            if l.startswith("#"):
                continue
            l = l.split()
            x,y,z = float(l[1]),float(l[2]),float(l[3])
            S = np.asarray([x,y,z])
            r11, r12, r13, r21, r22, r23, r31, r32, r33 = [float(l[i]) for i in range(7,16)]
            R = np.array([
                [r11, r12, r13],
                [r21, r22, r23],
                [r31, r32, r33],
            ])
            out[l[0]] = [S,R]
            
    return out


@njit
def _radius_pts(distance):
    """
    Determine le rayon d'un point projeté sur la depthmap en fonction de sa 
    distance au point de vue
    """
    
    if distance<1:
        return 2
    elif distance<2.5:
        return 2 
    elif distance<4:
        return 1
    else:
        return 0


@njit
def _spherical_projection(xyz, indexation, center_coordinates, resolution_y=600):
    """
    Génère une image sphérique panoramique (360°) depuis un point de vue.
    
    Le nuage est projeté en coordonnées sphériques (theta, phi) puis converti
    en coordonnées image. Les points sont triés par distance décroissante pour 
    gérer l'occlusion (les points proches écrasent les points lointains).
    
    Args:
        xyz                 : np.ndarray (n, 3) nuage de points xyz
        indexation          : np.ndarray (n,) indexation du nuage de points
        center_coordinates  : np.ndarray (3,) position du point de vue (x, y, z)
        resolution_y        : résolution verticale en pixels (horizontale = 2x)
                              défaut 600 -> image 1200 x 600
    
    Returns:
        mapping   : np.ndarray (H, W) int32
                    indice du point (pc.indexation) projeté sur chaque pixel
                    -1 si aucun point
        depth     : np.ndarray (H, W) float64
                    distance au point de vue par pixel
                    np.inf si aucun point
        position  : np.ndarray (H, W, 3) float64
                    coordonnées 3D (x, y, z) du point projeté par pixel
                    np.nan si aucun point
        depth_occ : np.ndarray (H, W) float64
                    carte de profondeur avec rayon dynamique (radius_pts)
                    utilisée pour la détection d'occlusion
                    np.inf si aucun point
    
    Notes:
        - Système de coordonnées sphériques :
            theta : angle azimutal [0, 2π] -> axe horizontal image
            phi   : angle zénithal [0, π]  -> axe vertical image
        - Le rayon dynamique (radius_pts) agrandit les points proches
          pour compenser la densité décroissante avec la distance
    """
    
    # centrage du PointCloud
    translated_points = xyz - center_coordinates
    # distances = np.linalg.norm(translated_points, axis=1)
    distances = np.sqrt((translated_points ** 2).sum(axis=1))

    norms = np.argsort(distances)[::-1]
    
    # Trie des points par distance au centre
    translated_points = translated_points[norms]
    sorted_point_cloud = xyz[norms]
    id_pts = indexation[norms]
    distances = distances[norms]
    
    # Passage en coord sphérique
    theta = (np.arctan2(translated_points[:, 1], translated_points[:, 0]) + 0.5 * np.pi) % (2 * np.pi)
    phi = np.arccos(translated_points[:, 2] / distances)
    
    # Convserion en coord images (pixels)
    resolution_x = 2 * resolution_y
    x = (theta / (2 * np.pi) * resolution_x).astype(np.int32)
    y = (phi / np.pi * resolution_y).astype(np.int32)

    # Création des rasters
    mapping = -1 * np.ones((resolution_y, resolution_x), dtype=np.int32)
    depth = np.inf * np.ones((resolution_y, resolution_x), dtype=np.float64)
    position = np.full((resolution_y, resolution_x, 3), np.nan, dtype=np.float64)
    depth_occ = np.full((resolution_y, resolution_x), np.inf, dtype=np.float64)

    # Remplissage rasters
    for i in range(len(translated_points)):
        ix = min(max(x[i], 0), resolution_x - 1)
        iy = min(max(y[i], 0), resolution_y - 1)
        d = distances[i]
        r = _radius_pts(d)
        if d < depth[iy, ix]:
            depth[iy, ix] = d
            mapping[iy, ix] = id_pts[i]
            position[iy, ix, :] = sorted_point_cloud[i]
            
            # Carte de profondeur avec rayon dynamique (radius_pts())
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx = ix + dx
                    ny = iy + dy
                    if 0 <= nx < resolution_x and 0 <= ny < resolution_y:
                        # if d < depth[ny, nx]:
                        depth_occ[ny, nx] = d

    return mapping, depth, position, depth_occ


def spherical_projection(pc: PointCloud, center_coordinates: np.ndarray, resolution_y: int = 600):
    """
    Wrapper PointCloud pour _spherical_projection.
    
    Args:
        pc                 : PointCloud source — doit être indexé
        center_coordinates : np.ndarray (3,) position du point de vue
        resolution_y       : résolution verticale en pixels (horizontale = 2x)
    
    Returns:
        mapping, depth, position, depth_occ
    """
    if not pc.is_indexed():
        raise ValueError("PointCloud non indexé — appeler pc.index() avant")
    
    return _spherical_projection(
        xyz                = pc.xyz.astype(np.float64),
        indexation         = pc.indexation.astype(np.int32),
        center_coordinates = center_coordinates.astype(np.float64),
        resolution_y       = resolution_y,
    )


def fill_map(array: np.ndarray, method: str = 'nearest'):
    """
    Comble les valeurs manquantes (NaN) d'une map par interpolation.
    
    Args:
        array  : np.ndarray (H, W) ou (H, W, C)
        method : méthode d'interpolation 'nearest' | 'linear' | 'cubic'
    
    Returns:
        array interpolé, même shape que l'entrée
    """
    li, co = array.shape[:2]
    grid_x, grid_y = np.mgrid[0:li:li*1j, 0:co:co*1j]

    # Points valides (non NaN)
    isnan  = np.isnan(array if array.ndim == 2 else array[:, :, 0])
    points = np.argwhere(~isnan)

    if array.ndim == 2:
        values = array[tuple(points.T)]
        return griddata(points, values, (grid_x, grid_y), method=method)

    # Multi-canaux
    array_interp = np.zeros_like(array)
    for c in range(array.shape[2]):
        values = array[:, :, c][tuple(points.T)]
        array_interp[:, :, c] = griddata(points, values, (grid_x, grid_y), method=method)

    return array_interp


@njit(parallel=True)  # pas de fastmath pour préserver le résultat
def equirectangular_transform_fast(image, R):
    """
    Réalise une rotation d'image equirectangulaire à partir de l'orientation
    au format OPK.
    
    Args:
        image : np.ndarray (H, W, 3) image panoramique
        R     : np.ndarray (3, 3) matrice de rotation issu du OPK
    
    Returns:
        output : image panoramique avec correction de l'orientation (Nord au centre)
    """
    height, width = image.shape[:2]

    # Matrices identiques à ta version
    yaw = np.radians(90.0)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]])

    Rcp = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0,-1.0, 0.0]])

    Rmc = R.T
    Rmp = Rz @ Rmc @ Rcp  # produit à droite dans ta version NumPy

    output = np.zeros_like(image)

    # Éviter division par zéro si width/height == 1 (cas pathologique)
    wden = 1.0 if width  == 1 else (width  - 1.0)
    hden = 1.0 if height == 1 else (height - 1.0)

    for j in prange(height):
        theta = (j / hden) * np.pi  # identique à linspace(0, pi, height)
        st = np.sin(theta)
        ct = np.cos(theta)

        for i in range(width):
            phi = (i / wden) * 2.0 * np.pi  # identique à linspace(0, 2pi, width)
            cp = np.cos(phi)
            sp = np.sin(phi)

            # Coord. cartésiennes
            x = st * cp
            y = st * sp
            z = ct

            # ----- PRODUIT A DROITE: [x,y,z] @ Rmp -----
            # i.e. combinaison des colonnes de Rmp, pas des lignes !
            xr = x*Rmp[0,0] + y*Rmp[1,0] + z*Rmp[2,0]
            yr = x*Rmp[0,1] + y*Rmp[1,1] + z*Rmp[2,1]
            zr = x*Rmp[0,2] + y*Rmp[1,2] + z*Rmp[2,2]

            # Back to spherical
            phi_rot = np.arctan2(yr, xr)   # (-pi, pi]
            # clamp numérique pour éviter arccos hors domaine à cause des arrondis
            if zr > 1.0:  zr = 1.0
            if zr < -1.0: zr = -1.0
            theta_rot = np.arccos(zr)      # [0, pi]

            # Mapping pixels (même logique: troncature vers zéro puis modulo)
            uu = int((phi_rot / (2.0*np.pi)) * width)
            vv = int((theta_rot / np.pi) * height)
            u = uu % width
            v = vv % height

            output[j, i] = image[v, u]

    return output 


def reshape_borders(contours: tuple, hierarchy: np.ndarray, seuil: int = 12):
    """
    Sépare et restructure les contours OpenCV en contours extérieurs et intérieurs.
    
    Args:
        contours  : contours retournés par cv2.findContours
        hierarchy : hiérarchie retournée par cv2.findContours (RETR_CCOMP)
        seuil     : nombre minimum de points pour qu'un contour soit conservé
    
    Returns:
        contours_ext : list of np.ndarray (n, 2) — contours extérieurs
        contours_int : list of list of np.ndarray (n, 2) — contours intérieurs
                       groupés par contour extérieur parent
    """
    h = hierarchy[0]
    contours_ext = []
    contours_int = []

    def reshape(contour):
        """Supprime la dimension inutile (n, 1, 2) -> (n, 2)."""
        return contour.reshape(-1, 2)

    for i in range(len(h)):
        parent = h[i][3]

        # Contours extérieurs uniquement (pas de parent)
        if parent != -1:
            continue

        contour = reshape(contours[i])

        # Filtrage par taille
        if len(contour) <= seuil:
            continue

        contours_ext.append(contour)

        # Contours intérieurs (enfants)
        child_list = []
        fchild = h[i][2]  # premier enfant
        while fchild != -1:
            child_list.append(reshape(contours[fchild]))
            fchild = h[fchild][0]  # enfant suivant

        contours_int.append(child_list)

    return contours_ext, contours_int





