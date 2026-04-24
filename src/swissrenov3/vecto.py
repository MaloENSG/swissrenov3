# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:54:33 2025

@author: malo.delacour
"""

import numpy as np

import shapely
from shapely.geometry import LineString, Point
from shapely import MultiLineString
from shapely import Polygon
from shapely.geometry import shape
import json


def inter_pt(x1, y1, x2, y2, t):
    xp = x1 + t * (x2 - x1)
    yp = y1 + t * (y2 - y1)
    return xp, yp


def polyline2linestring(polylines):
    """
    Converti plusieurs polylignes sous forme d'array n*2 (x, y) en une liste
    de segments simples 2*2 de type LineString

    Parameters
    ----------
    polylines : list de np array (n*2)
        Liste contenant les arrays.

    Returns
    -------
    segments : list de LineString
        Liste de segments simples de type LineString.

    """
    segments = []
    for polyline in polylines:
        segments += [LineString([polyline[i], polyline[i + 1]]) for i in range(len(polyline) - 1)]
    return segments


def polyline_visible(position, segments, distance=21):
    """
    Retourne les éléments visibles depuis le point de vu. Permet de retirer
    les éléments masqué par d'autres.

    Parameters
    ----------
    position : array (2,)
        Position X, Y du point de vue.
    segments : list de LineString Simple
        Liste des segments .
    distance : int or float, optional
        Distance de détection des obstacles.
        Les obstacles plus loin ne sont pas considéré. The default is 16.

    Returns
    -------
    viewed : list de LineString
        Liste de LineString simple (2*2) des segments ou morceaux de segments
        conservé.
    viewed_state : list de boolean
        Liste de l'état de chaque segment.
        True -> Segment complet.
        False -> Segment partiel.

    """

    viewed = []
    viewed_state = []
    pt = [tuple(position)]
    shape_segments = MultiLineString(segments)
    
    for seg in segments:
        
        ###
        # Création du triangle de test
        line = list(seg.coords)
        # pt = list(viewpoint_geom.coords)
        polygon = Polygon(line+pt)
        line_dilated = seg.buffer(0.01)
        
        # Test du triangle
        polygon = polygon - line_dilated    
        res = shapely.intersection(polygon, shape_segments)
        
        # Si occlusion
        if res.is_empty == False:
            
            ###
            # Conversion des obstacles en liste de segments simples
            liste_obstacle = []
            if res.geom_type == 'LineString':
                obstacle = np.asarray(res.coords)
                liste_obstacle += [LineString([obstacle[0], obstacle[1]])]
                 
            if res.geom_type == 'MultiLineString':
                for linestring in list(res.geoms):
                    obstacle = np.asarray(linestring.coords)
                    liste_obstacle += [LineString([obstacle[0], obstacle[1]])]
                    
            ###
            # Calcul de la projection de l'obstacle sur le segment
            # Pour chaque obstacle
            for obstacle in liste_obstacle:
                obstacle_array = np.asarray(obstacle.coords)
                x1, y1, x2, y2 = obstacle_array[0,0], obstacle_array[0, 1], obstacle_array[1, 0], obstacle_array[1, 1]
                
                # Projection pt1
                dist_pt = np.sqrt((position[0]-x1)**2 + (position[1]-y1)**2)
                t = distance/dist_pt
                X1, Y1 = inter_pt(position[0], position[1], x1, y1, t)
                
                # Projection pt2
                dist_pt = np.sqrt((position[0]-x2)**2 + (position[1]-y2)**2)
                t = distance/dist_pt
                X2, Y2 = inter_pt(position[0], position[1], x2, y2, t)
                
                # Soustaction de la partie caché
                occ_polygon = Polygon([[X1, Y1], [X2, Y2], position]) # Construction du polygon d'occlusion
                occ_polygon = occ_polygon.buffer(0.01)                # Dilatation pour éviter des effets de bords
                seg = seg - occ_polygon                               # Difference du segment par le polygon d'occlusion
            
            ###
            # Si une partie du segment est visible
            if seg.is_empty == False:
                # Segment unique
                if seg.geom_type == 'LineString':
                    viewed.append(seg)
                    viewed_state.append(False)
                
                # Segment multiple
                if seg.geom_type == 'MultiLineString':
                    for linestring in list(seg.geoms):
                        viewed.append(linestring)
                        viewed_state.append(False)
        
        # Si pas d'occlusion
        else :
            viewed.append(seg)
            viewed_state.append(True)
            
    return viewed, viewed_state


def linestring2numpy(segments):
    segments_listarray = []
    for line in segments:
        segments_listarray.append(np.asarray(line.coords))
    return segments_listarray


def xy2YawPitch(segments, position, h=0):
    
    sphere_segments = []
    for seg in segments:
        
        alti = np.array([[h], [h]])
        segment = np.hstack((seg, alti))
        
        translated_segment = segment - position
        theta = np.arctan2(translated_segment[:, 1], translated_segment[:, 0])            # yaw + pi
        theta = theta - np.pi/2
        phi = -np.arccos(translated_segment[:, 2] / np.linalg.norm(translated_segment, axis=1)) +np.pi/2# pitch
        
        sphere_segment = np.vstack((theta, phi))
        # yaw1   , yaw2
        # pitch1 , pitch2
        sphere_segments.append(sphere_segment.T)
        
    return sphere_segments


def json2multilinestring(geojson_file):
    """
    Retourne un MultiLineString contenant des LineString simples (2*2) à partir
    d'un fichier geojson'

    Parameters
    ----------
    geojson_file : str
        Chemin du fichier geojson.

    Returns
    -------
    lines : MultiLineString
        Objet MultiLineString à convertir en list de LineString simples.

    """

    with open(geojson_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    
    lines = MultiLineString([])
    for element in data['features']:
        geom = shape(element['geometry'])
        geom_array = np.asarray(geom.geoms[0].coords)
        geom = MultiLineString([LineString([geom_array[i], geom_array[i + 1]]) for i in range(len(geom_array) - 1)])
        lines = lines | geom # union
    
    return lines


