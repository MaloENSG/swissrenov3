# Structure des fichiers — Nuages de points

Cette page décrit l'organisation des nuages de points pour une friche.

---

## Vue d'ensemble

```
friche/
├── Station/
│   └── YYYY-MM-DD/
│       ├── station1.las
│       ├── station1.pts
│       ├── station1.e57
│       └── ...
└── Pointcloud/
    └── id_pointcloud/
        ├── id_pointcloud.las
        ├── semantique.json
        ├── metadata.json
        └── Visualisation/
            ├── metadata_viz.json
            ├── etage1/
            └── etage2/
                └── piece21/
                    └── id_pointcloud_room_21/
```

---

## Description des dossiers et fichiers

### `Station/`

Données brutes issues des scans LiDAR, organisées par date d'acquisition. Chaque sous-dossier correspond à une époque d'acquisition.

---

### `YYYY-MM-DD/`

Session datée au format YYYY-MM-DD (ex : `2024-03-15`), garantissant un tri chronologique.

Chaque station de scan est représentée par un fichier dans l'un des formats suivants :

| Format | Extension | Description |
|---|---|---|
| LAS | `.las` | Format binaire standard pour nuages de points, avec attributs (intensité, classification, RGB…) |
| PTS | `.pts` | Format texte simple : X Y Z Intensité [R G B] par ligne |
| E57 | `.e57` | Format ouvert d'échange de données 3D, supportant les images sphériques associées |

> Un même scan peut être disponible dans plusieurs formats selon les besoins d'interopérabilité.

---

### `Pointcloud/`

Nuages de points traités et fusionnés, produits à partir des scans bruts de `Station/`. Chaque sous-dossier correspond à un nuage de points consolidé et identifié.

---

### `nom_pointcloud/`

Dossier d'un nuage de points traité, identifié par un nom unique de la forme `Friche_batiment_densite_NN` (ex : `condor_A_10_01`).

| Fichier | Format | Description |
|---|---|---|
| `nom_pointcloud.las` | LAS | Nuage de points fusionné et échantillonnées (dans le référentiel monde ou non). |
| `semantique.json` | JSON | Classification sémantique de points pour une classe spécifique et limité. |
| `metadata.json` | JSON | Métadonnées du nuage : paramètres de traitement, CRS, nombre de points, stations sources etc. (**Structure à préciser**). |

---

#### `Visualisation/`

Données dérivées du nuage de points, organisées pour la visualisation interactive par niveau et par pièce au format potree.

| Fichier | Description |
|---|---|
| `metadata_viz.json` | Paramètres de la visualisation : niveaux disponibles, pièces indexées. |

La hiérarchie spatiale est structurée en **étages** puis en **pièces** :

```
Visualisation/
├── metadata_viz.json
├── etage1/
└── etage2/
    └── piece21/
        └── id_pointcloud_room_21/
```

##### `etageN/`

Dossier regroupant les données de visualisation d'un niveau du bâtiment. Contient autant de sous-dossiers que de pièces identifiées à cet étage.

##### `etageN/pieceNN/`

Dossier d'une pièce spécifique. Contient le sous-nuage de points découpé et indexé pour cette pièce.

##### `etageN/pieceNN/id_pointcloud_room_NN/`

Nuage de points de la pièce, prêt pour le rendu. Le nom reprend l'identifiant du nuage parent suivi du numéro de pièce pour garantir la traçabilité.

---

## Conventions de nommage

| Élément | Convention | Exemple |
|---|---|---|
| Session | `YYYY-MM-DD` | `2024-03-15` |
| Fichier station | `stationN.[ext]` | `station3.las` |
| Identifiant nuage | `pc_[site]_[zone]_NN` | `pc_friche_nord_01` |
| Dossier étage | `etageN` | `etage2` |
| Dossier pièce | `pieceNN` | `piece21` |
| Nuage de pièce | `[id_pointcloud]_room_NN` | `pc_friche_nord_01_room_21` |

