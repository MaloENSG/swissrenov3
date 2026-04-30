"""
Conversion de meshes .obj (sol, toit, mur, solid) en fichier gbXML valide.

Le point de départ de chaque polygone est choisi à la transition
bord-bas → bord-haut, ce qui garantit que les viewers gbXML
(qui tesselent en éventail) produisent une géométrie correcte,
même pour les polygones concaves à profil en escalier.

Usage :
    python obj_to_gbxml.py solid.obj sol.obj toit.obj mur.obj output.gbxml
"""

import trimesh
import numpy as np
import xgbxml
from collections import defaultdict


# ---------------------------------------------------------------------------
# Géométrie : utilitaires polygon 3D
# ---------------------------------------------------------------------------

def _newell_normal(pts: np.ndarray) -> np.ndarray:
    """
    Calcule la normale d'un polygone par la méthode de Newell.
    Robuste pour les polygones non-triangulaires et légèrement non-planaires.
    """
    n = len(pts)
    nx = ny = nz = 0.0
    for i in range(n):
        c = pts[i]; nx_ = pts[(i + 1) % n]
        nx += (c[1] - nx_[1]) * (c[2] + nx_[2])
        ny += (c[2] - nx_[2]) * (c[0] + nx_[0])
        nz += (c[0] - nx_[0]) * (c[1] + nx_[1])
    v = np.array([nx, ny, nz])
    L = np.linalg.norm(v)
    return v / L if L > 1e-12 else v


def order_boundary_edges(boundary_edges: np.ndarray) -> list[int]:
    """
    Ordonne les arêtes de bord d'une facette en une boucle fermée.
    Lève ValueError si le bord est non-manifold (degré ≠ 2).
    """
    graph: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        graph[a].append(b)
        graph[b].append(a)

    bad = [v for v, nb in graph.items() if len(nb) != 2]
    if bad:
        raise ValueError(f"Bord non-manifold : {len(bad)} sommet(s) avec degré ≠ 2.")

    start = int(boundary_edges[0][0])
    ordered = [start]
    prev, current = None, start

    while True:
        nb = graph[current]
        nxt = nb[0] if nb[0] != prev else nb[1]
        if nxt == start:
            break
        ordered.append(nxt)
        prev, current = current, nxt

    return ordered


def remove_colinear_points(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Supprime les points colinéaires dans un polygone 3D.
    La tolérance porte sur le produit vectoriel des directions normalisées,
    indépendamment de la longueur des arêtes.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return pts
    keep = []
    for i in range(n):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[(i + 1) % n] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        if np.linalg.norm(np.cross(v1 / n1, v2 / n2)) > tol:
            keep.append(i)
    return pts[keep] if keep else pts


def ensure_correct_winding(contour: np.ndarray, reference_normal: np.ndarray) -> np.ndarray:
    """
    Oriente le contour selon la normale de référence (right-hand rule).
    Si la normale de Newell est opposée, les points sont inversés
    en conservant le premier point.
    """
    if len(contour) < 3:
        return contour
    if np.dot(_newell_normal(contour), reference_normal) < 0:
        return np.concatenate([contour[:1], contour[1:][::-1]])
    return contour


def reorder_from_transition(contour: np.ndarray) -> np.ndarray:
    """
    Réordonne le contour pour que le point de départ soit à la première
    transition bas → haut dans le sens de parcours du polygone.

    Logique :
    - "bas" = Z < seuil (médiane des Z), "haut" = Z >= seuil
    - On cherche l'indice i tel que le point i est "bas" et le point i+1 est "haut"
      (première montée dans l'ordre circulaire)
    - Si le polygone est entièrement bas ou entièrement haut (toit/sol plat),
      on tombe en fallback : Z min, puis X min, puis Y min

    Ce choix garantit que le polygone commence sur le bord bas juste avant
    de monter, ce que les viewers gbXML tesselent correctement.
    """
    n = len(contour)
    if n < 3:
        return contour

    z = contour[:, 2]
    z_threshold = np.median(z)

    is_low = z < z_threshold

    # Cas dégénéré : tous au même niveau (face plane horizontale ou verticale uniforme)
    if is_low.all() or (~is_low).all():
        idx = np.lexsort((contour[:, 1], contour[:, 0], contour[:, 2]))
        return np.roll(contour, -idx[0], axis=0)

    # Chercher la première transition bas → haut
    for i in range(n):
        if is_low[i] and not is_low[(i + 1) % n]:
            # i est le dernier point bas avant la montée → on commence en i+1
            # pour avoir le premier point haut en tête... non, on veut commencer
            # sur le dernier bas (juste avant la montée) pour que le contour
            # parte du coin bas-gauche du mur
            return np.roll(contour, -i, axis=0)

    # Fallback : pas de transition trouvée (ne devrait pas arriver)
    idx = np.lexsort((contour[:, 1], contour[:, 0], contour[:, 2]))
    return np.roll(contour, -idx[0], axis=0)


# ---------------------------------------------------------------------------
# RectangularGeometry : tilt, azimuth, width, height, origin
# ---------------------------------------------------------------------------
 
def compute_tilt_azimuth(normal: np.ndarray) -> tuple[float, float]:
    """
    Calcule le tilt et l'azimuth d'une face à partir de sa normale unitaire.
 
    Tilt    : angle entre la normale et la verticale (Z).
              0°  = face horizontale vers le haut (toit plat)
              90° = face verticale (mur)
              180°= face horizontale vers le bas (sol)
 
    Azimuth : orientation de la projection horizontale de la normale,
              convention gbXML boussole : 0°=Nord, 90°=Est, sens horaire.
              Non significatif pour les faces horizontales (tilt=0 ou 180).
 
    Args:
        normal : vecteur normal unitaire (3,) de la face
 
    Returns:
        (tilt_deg, azimuth_deg)
    """
    # Tilt : angle entre la normale et l'axe Z
    # Clamp pour éviter les erreurs numériques sur arccos
    cos_tilt = np.clip(normal[2], -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_tilt))
 
    # Azimuth : projection de la normale sur le plan horizontal
    # Convention math → boussole : (90 - arctan2(y, x)) % 360
    az_math = np.degrees(np.arctan2(normal[1], normal[0]))
    azimuth_deg = (90.0 - az_math) % 360.0
 
    # Pour les faces horizontales (tilt ≈ 0 ou ≈ 180), l'azimuth n'est pas
    # significatif : on le fixe à 0 par convention gbXML
    if abs(tilt_deg) < 1e-3 or abs(tilt_deg - 180.0) < 1e-3:
        azimuth_deg = 0.0
 
    return tilt_deg, azimuth_deg
 
 
def compute_width_height(polyloop: np.ndarray) -> tuple[float, float]:
    """
    Calcule la largeur et la hauteur d'une face via sa bounding box orientée.
 
    La bounding box orientée (OBB) est calculée dans le plan de la face,
    ce qui donne les dimensions réelles indépendamment de l'orientation 3D.
 
    Pour un polygone plan, l'OBB a une dimension quasi-nulle (épaisseur) :
    on prend les deux autres dimensions triées par ordre décroissant.
 
    Args:
        polyloop : tableau (N, 3) des sommets du polygone
 
    Returns:
        (width, height) avec width ≥ height
    """
    transform, extents = trimesh.bounds.oriented_bounds(
        polyloop,
        angle_digits=1,
        ordered=True,
        normal=None,
        coplanar_tol=1e-12,
    )
    # extents triés : [épaisseur~0, petite_dim, grande_dim]
    extents_sorted = np.sort(extents)
    width  = extents_sorted[2]  # plus grande dimension
    height = extents_sorted[1]  # deuxième dimension
    return float(width), float(height)
 
 
def _add_rectangular_geometry(
    surface,
    contour: np.ndarray,
    normal:  np.ndarray,
) -> None:
    """
    Ajoute un élément RectangularGeometry à une surface gbXML.
 
    Contenu :
    - Azimuth   : orientation de la normale (convention boussole, degrés)
    - CartesianPoint : premier point du PolyLoop (point canonique de la face)
    - Tilt      : inclinaison de la face (degrés)
    - Width     : plus grande dimension de la bounding box orientée (m)
    - Height    : deuxième dimension (m)
 
    Args:
        surface : élément Surface xgbxml auquel ajouter la géométrie
        contour : tableau (N, 3) des sommets (premier point = origine)
        normal  : normale unitaire de la face
    """
    tilt, azimuth = compute_tilt_azimuth(normal)
    width, height = compute_width_height(contour)
    origin = contour[0]  # premier point = point canonique (reorder_from_transition)
 
    rect_geo = surface.add_RectangularGeometry()
 
    rect_geo.add_Azimuth().text  = f"{azimuth:.4f}"
 
    cp = rect_geo.add_CartesianPoint()
    cp.add_Coordinate().text = f"{origin[0]:.6f}"
    cp.add_Coordinate().text = f"{origin[1]:.6f}"
    cp.add_Coordinate().text = f"{origin[2]:.6f}"
 
    rect_geo.add_Tilt().text   = f"{tilt:.4f}"
    rect_geo.add_Height().text = f"{height:.6f}"
    rect_geo.add_Width().text  = f"{width:.6f}"



# ---------------------------------------------------------------------------
# Extraction des contours depuis un Trimesh
# ---------------------------------------------------------------------------

def extract_contours(mesh: trimesh.Trimesh) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extrait les contours (polygones) de toutes les facettes du mesh.

    Retourne une liste de (contour, normale) :
    - contour : (N, 3) ordonné, nettoyé, orienté, point de départ à la transition bas→haut
    - normale : normale unitaire de la facette (depuis les normales du .obj)
    """
    mesh.merge_vertices()
    results: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0

    for i, facet in enumerate(mesh.facets):
        boundary_edges = mesh.facets_boundary[i]
        if boundary_edges.shape[0] == 0:
            continue
        try:
            ordered_indices = order_boundary_edges(boundary_edges)
        except ValueError as exc:
            print(f"[WARN] Facette {i} ignorée : {exc}")
            skipped += 1
            continue

        contour = mesh.vertices[ordered_indices]
        contour = remove_colinear_points(contour)
        if len(contour) < 3:
            continue

        facet_normal = mesh.face_normals[facet].mean(axis=0)
        norm = np.linalg.norm(facet_normal)
        if norm > 1e-12:
            facet_normal /= norm

        contour = ensure_correct_winding(contour, facet_normal)
        contour = reorder_from_transition(contour)
        results.append((contour, facet_normal))

    if skipped:
        print(f"[INFO] {skipped} facette(s) ignorée(s) (non-manifold).")

    # Triangles isolés (non regroupés en facette coplanaire)
    all_faces   = set(range(len(mesh.faces)))
    facet_faces = set(np.concatenate(mesh.facets)) if len(mesh.facets) > 0 else set()

    for face_id in all_faces - facet_faces:
        verts       = mesh.vertices[mesh.faces[face_id]]
        face_normal = mesh.face_normals[face_id]
        verts = ensure_correct_winding(verts, face_normal)
        verts = reorder_from_transition(verts)
        results.append((verts, face_normal))

    return results


# ---------------------------------------------------------------------------
# Export gbXML
# ---------------------------------------------------------------------------

SURFACE_TYPE_MAP = {
    "ground": "SlabOnGrade",
    "roof":   "Roof",
    "wall":   "ExteriorWall",
    "shade":  "Shade",
}


def _add_poly_loop(parent, contour: np.ndarray) -> None:
    """Ajoute un PolyLoop avec ses CartesianPoints à un élément xgbxml."""
    poly_loop = parent.add_PolyLoop()
    for pt in contour:
        cp = poly_loop.add_CartesianPoint()
        cp.add_Coordinate().text = f"{pt[0]:.6f}"
        cp.add_Coordinate().text = f"{pt[1]:.6f}"
        cp.add_Coordinate().text = f"{pt[2]:.6f}"

  

def _add_surfaces(
    campus,
    contours:     list[tuple[np.ndarray, np.ndarray]],
    surface_type: str,
    name_prefix:  str,
    id_counter:   list[int],
) -> None:
    """Ajoute les surfaces au campus avec ids uniques et bon surfaceType."""
    gbxml_type = SURFACE_TYPE_MAP[surface_type]
    for i, (contour, normal) in enumerate(contours):
        sid = id_counter[0]
        id_counter[0] += 1
        surface = campus.add_Surface(id=f"Surface-{sid}", surfaceType=gbxml_type)
        surface.add_Name().text = f"{name_prefix}_{i}"
        
        if surface_type != "shade" :
            surface.add_AdjacentSpaceId(spaceIdRef="Space-1")
        
        _add_poly_loop(surface.add_PlanarGeometry(), contour)
        _add_rectangular_geometry(surface, contour, normal)


def make_gbxml(
    solid_contours:  list[tuple[np.ndarray, np.ndarray]],
    ground_contours: list[tuple[np.ndarray, np.ndarray]],
    roof_contours:   list[tuple[np.ndarray, np.ndarray]],
    wall_contours:   list[tuple[np.ndarray, np.ndarray]],
    shade_contours:  list[tuple[np.ndarray, np.ndarray]],
    out_path: str,
) -> None:
    """
    Génère un fichier gbXML valide.

    Structure :
      Campus
        Building
          Space-1
            ShellGeometry  ← solid_contours (enveloppe fermée)
        Surface-N          ← ground / roof / wall
    """
    gbxml = xgbxml.create_gbXML()

    campus   = gbxml.add_Campus(id="Campus-1")
    building = campus.add_Building(id="Building-1")
    building.add_Name().text = "Batiment"

    space = building.add_Space(id="Space-1")
    space.add_Name().text = "Espace_1"

    shell_geo    = space.add_ShellGeometry()
    closed_shell = shell_geo.add_ClosedShell()
    for contour, _ in solid_contours:
        _add_poly_loop(closed_shell, contour)

    id_counter = [1]
    _add_surfaces(campus, ground_contours, "ground", "Sol",  id_counter)
    _add_surfaces(campus, roof_contours,   "roof",   "Toit", id_counter)
    _add_surfaces(campus, wall_contours,   "wall",   "Mur",  id_counter)
    _add_surfaces(campus, shade_contours,   "shade", "Ombre", id_counter)

    tree = gbxml.getroottree()
    tree.write(out_path, pretty_print=True)
    print(f"[OK] gbXML écrit : {out_path}")
    print(
        f"     ShellGeometry : {len(solid_contours)} polygone(s)\n"
        f"     Surfaces      : {len(ground_contours)} sol, "
        f"{len(roof_contours)} toit, {len(wall_contours)} mur"
    )


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def load_mesh(path: str) -> trimesh.Trimesh:
    """Charge un .obj et vérifie qu'on obtient bien un Trimesh."""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"'{path}' n'a pas produit un Trimesh (type : {type(mesh).__name__})."
        )
    print(f"[INFO] {path} — {len(mesh.faces)} faces, {len(mesh.vertices)} sommets")
    return mesh


def obj_to_gbxml(
    solid_path:  str,
    ground_path: str,
    roof_path:   str,
    wall_path:   str,
    shade_path:  str,
    output_path: str,
) -> None:
    """
    Convertit 4 fichiers .obj en un fichier gbXML valide.

    Args:
        solid_path  : enveloppe fermée du bâtiment (→ ShellGeometry du Space)
        ground_path : faces sol  (→ Surface UndergroundSlab)
        roof_path   : faces toit (→ Surface Roof)
        wall_path   : faces mur  (→ Surface ExteriorWall)
        output_path : chemin du fichier gbXML produit
    """
    solid_contours  = extract_contours(load_mesh(solid_path))
    ground_contours = extract_contours(load_mesh(ground_path))
    roof_contours   = extract_contours(load_mesh(roof_path))
    wall_contours   = extract_contours(load_mesh(wall_path))
    shade_contours  = extract_contours(load_mesh(shade_path))

    print(
        f"[INFO] Contours extraits — "
        f"Solid : {len(solid_contours)}, "
        f"Sol : {len(ground_contours)}, "
        f"Toit : {len(roof_contours)}, "
        f"Mur : {len(wall_contours)}"
    )

    make_gbxml(solid_contours, ground_contours, roof_contours, wall_contours, shade_contours, output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    frichename = "FR"
    EGIDname = "190195516"
    
    obj_to_gbxml(
        solid_path  = f"res/{frichename}/bldg_closed_{EGIDname}.obj",
        ground_path = f"res/{frichename}/bldg_ground_{EGIDname}.obj",
        roof_path   = f"res/{frichename}/bldg_int_{EGIDname}.obj",
        wall_path   = f"res/{frichename}/bldg_wall_{EGIDname}.obj",
        shade_path  = f"res/{frichename}/bldg_ext_{EGIDname}.obj",
        output_path = f"res/{frichename}/bldg_{EGIDname}.xml",
    )

