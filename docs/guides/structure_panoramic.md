# Structure des fichiers — Panoramiques

Cette page décrit l'organisation des images panoramiques produites par le pipeline d'acquisition et de traitement pour une friche.

---

## Vue d'ensemble

```
friche/
└── Panoramiques/
    └── YYYY-MM-DD/
        └── Sequence_00/
            ├── metadata.json
            └── pano_0000/
                ├── 2D_SAM/
                │   ├── pano_0000_segments.jpg
                │   └── pano_0000_segments.tif
                ├── 3D_fusion/
                │   ├── pano_0000depth.tif
                │   ├── pano_0000mapping.tif
                │   ├── pano_0000position.tif
                │   └── metadata_3dmap.json
                ├── pano_0000_cubemap/
                │   ├── face_0.jpg
                │   └── ...
                ├── pano_0000_tuiles/
                │   ├── tuile_0.webp
                │   └── ...
                ├── pano_0000_HD.jpg
                └── pano_0000_SD.webp
```

---

## Description des dossiers et fichiers

### `Panoramiques/`

Dossier principal regroupant toutes les sessions de capture panoramique. Chaque sous-dossier correspond à une journée d'acquisition terrain.

---

### `YYYY-MM-DD/`

Session datée. Le format de nommage est YYYY-MM-DD (ex : `2024-04-24`) pour garantir un tri chronologique.

---

### `Sequence_00/`

Séquence de capture numérotée au sein d'une session. Plusieurs séquences sont acquisent durant une journée. Chaque séquence intègre des métadonnées dans le fichier `metadata.json`.

#### `metadata.json`

Métadonnées globales de la séquence. Généré automatiquement après traitement.

**Structure du fichier :**

```json
{
  "sequence_info": {
    "sequencename": "GS018174",
    "date_acquisition": "2025-07-08",
    "camera": "gopro max",
    "resolution": "5440*2720",
    "date_traitement": "2025-11-05"
  },
  "3Dmap": ["01"],
  "pano_list": [
    {
      "panoname": "pano_0045",
      "position": [2500000.00, 1250000.00, 400.00],
      "orientation": [
        [-0.0217, 0.9991, -0.0376],
        [ 0.1176, 0.0399,  0.9923],
        [ 0.9928, 0.0172, -0.1183]
      ]
    }
  ]
}
```

**Champs `sequence_info` :**

| Champ | Description |
|---|---|
| `sequencename` | Identifiant de la séquence |
| `date_acquisition` | Date de la prise de vue sur le terrain |
| `camera` | Modèle de caméra utilisé |
| `resolution` | Résolution en pixels (largeur × hauteur) |
| `date_traitement` | Date du traitement automatisé |

**Champs par panoramique (`pano_list`) :**

| Champ | Type | Description |
|---|---|---|
| `panoname` | string | Identifiant du panoramique, correspond au nom du dossier |
| `position` | array[3] | Coordonnées XYZ en **CH1903+/LV95** (Est, Nord, Altitude en mètres) |
| `orientation` | array[3][3] | Matrice de rotation 3×3 exprimant l'orientation de la caméra dans le référentiel monde |

**Champs (`3Dmap`) :**

Liste des identifiants de nuages de points 3D associés à cette séquence.

---

### `pano_0000/`

Panoramique individuel numéroté (ex : `pano_0000`, `pano_0001`, …). Chaque dossier regroupe l'ensemble des fichiers dérivés d'un seul point de vue.

#### Fichiers à la racine du panoramique

| Fichier | Format | Description |
|---|---|---|
| `pano_0000_HD.jpg` | JPEG | Image panoramique pleine résolution. |
| `pano_0000_SD.webp` | WebP | Version basse résolution pour prévisualisation rapide. |

---

#### `2D_SAM/`

Résultats de segmentation 2D produits par le modèle [SAM (Segment Anything)](https://segment-anything.com/). Contient les masques extraits de l'image panoramique.

| Fichier | Format | Description |
|---|---|---|
| `pano_0000_segments.jpg` | JPEG | Colorisation des segments pour la visualisation. |
| `pano_0000_segments.tif` | GeoTIFF | Sauvegarde efficace des masques de segments et de leur indice. Format de référence pour les traitements ultérieurs. |

---

#### `3D_fusion/`

Données issues de la reconstruction 3D : profondeur, correspondance d'indices et positionnement dans l'espace.

| Fichier | Format | Description |
|---|---|---|
| `pano_0000depth.tif` | GeoTIFF | Carte de profondeur estimée par reprojection. Valeurs encodées en mètres (flottant 32 bits). |
| `pano_0000mapping.tif` | GeoTIFF | Carte de correspondance pixel → point 3D. Utilisée pour une reprojection effices entre 2D et 3D. |
| `pano_0000position.tif` | GeoTIFF | Carte de position 3D relatif de chaque pixel dans un référentiel local. Il suffit d'appliquer l'offset dans `metadata_3dmap.json` pour passer en position absolue dans le référentiel monde. |
| `metadata_3dmap.json` | JSON | Paramètres et traçabilité de la fusion : transformations, système de coordonnées (CRS), offset du point de vue. |

---

#### `pano_0000_cubemap/`

Projection cubemap de la panoramique équirectangulaire. L'image est découpée en 6 faces carrées pour optenir des images similaire à des images perspectives. Cela permet de réaliser des détections avec des modèles d'IA dans les images sans déformations majeures.

`front, back, left, right, top, bottom`

---

#### `pano_0000_tuiles/`

Tuiles WebP pour affichage zoomable (format compatible [OSM](https://www.openstreetmap.org/)/[Leaflet](https://leafletjs.com/)). Permet le streaming progressif de la panoramique HD sans charger l'image entière.

> Par défaut, le tuilage produit 8 tuiles (2 lignes, 4 colonnes).

---

## Conventions de nommage (A REVOIR)

| Élément | Convention | Exemple |
|---|---|---|
| Session | `YYYY-MM-DD` | `2024-03-15` |
| Séquence | `Sequence_NN` (2 chiffres) | `Sequence_00` |
| Panoramique | `pano_NNNN` (4 chiffres) | `pano_0042` |
| Face cubemap | `face_N` (0 à 5) | `face_3` |
| Tuile | `tuile_N` | `tuile_12` |

