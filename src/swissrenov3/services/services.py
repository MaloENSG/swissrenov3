# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:22:12 2026

@author: malo
"""

import os
import json
from config import IFOLDER

import tifffile as tiff
import utils


def open_request(request):
    
    data = request.get_json()
    
    date = data['date']
    seqname = data['seqname']
    panoname = data['panoname']
    parameters = data['param']
    
    return date, seqname, panoname, parameters


def get_3d_coordinates(date, seqname, panoname, yaw, pitch):
    base_path = os.path.join(IFOLDER, seqname, f'pano_{panoname}', '3Dmap_01')

    # Chargement métadonnées
    with open(os.path.join(base_path, 'metadata_3dmap.json')) as f:
        data_3dmap = json.load(f)
    offset = data_3dmap['geo']['offset']

    # Chargement position
    position = tiff.imread(os.path.join(base_path, f'pano_{panoname}position.tif'))
    li, co, _ = position.shape

    x, y = utils.yawpitch2XY(yaw, pitch, co, li)
    X, Y, Z = position[int(y), co - int(x)]

    return {
        "X": round(X + offset[0], 3),
        "Y": round(Y + offset[1], 3),
        "Z": round(Z + offset[2], 3),
    }




