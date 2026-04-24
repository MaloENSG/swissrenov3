# struct classes todo

Les structures de données principales du projet sont organisées en quatre classes :
`PointCloudInfo`, `Referentiel`, `PointCloud` et `Raster`.

---

## 2.1 PointCloudInfo

Contient les métadonnées et informations de traçabilité associées à un nuage de points.
Cet objet est toujours présent dans un `PointCloud` via l'attribut `info`.

### Attributs

| Attribut | Type | Défaut | Description |
|---|---|---|---|
| `name` | `str` | `""` | Nom du nuage |
| `source` | `str` | `""` | Origine du fichier (chemin, instrument...) |
| `date` | `datetime` | now | Horodatage de création |
| `crs` | `str` | `""` | Système de coordonnées (ex: `"EPSG:2056"`) |
| `sampling` | `float` | `None` | Pas d'échantillonnage en mètres |
| `is_filtered` | `bool` | `False` | Indique si le nuage a été filtré |
| `is_indexed` | `bool` | `False` | Indique si le nuage a été indexé |
| `offset` | `list[float]` | `[0, 0, 0]` | Offset du référentiel local `[x, y, z]` |
| `angle` | `float` | `0.0` | Angle de rotation du référentiel local (degrés) |
| `scan_pos` | `np.ndarray` | `None` | Position du scanner `(x, y, z)` |
| `scan_rot` | `np.ndarray` | `None` | Rotation du scanner `(rx, ry, rz)` en radians |
| `extent_bbox` | `np.ndarray` | `None` | Boite englobante (4 ou 8 points) |
| `extent_poly` | `np.ndarray` | `None` | Contour 2D de l'emprise |

### Méthodes

#### `add_history(message)`

Ajoute une entrée horodatée dans l'historique des traitements.

```python linenums="1"
pc.info.add_history("Filtrage outliers statistiques")
pc.info.add_history("Reclassification sol")

print(pc.info.history)
# ['[2026-04-07 10:00:01] Filtrage outliers statistiques',
#  '[2026-04-07 10:01:23] Reclassification sol']
```

#### `history` *(property)*

Retourne la liste des entrées horodatées.

---

## 2.2 Referentiel

Définit un référentiel local par un offset `(x, y, z)` et un angle de rotation en Z.

### Attributs

| Attribut | Type | Défaut | Description |
|---|---|---|---|
| `x` | `float` | `0.0` | Offset en X (mètres) |
| `y` | `float` | `0.0` | Offset en Y (mètres) |
| `z` | `float` | `0.0` | Offset en Z (mètres) |
| `a` | `float` | `0.0` | Angle de rotation en Z (radians) |

### Properties

#### `offset`

Retourne l'offset sous forme de `np.ndarray (3,)`.

```python
ref = Referentiel(x=10.5, y=3.2, z=0.0, a=np.radians(45))
print(ref.offset)
# [10.5  3.2  0. ]
```

### Méthodes

#### `is_default()`

Retourne `True` si le référentiel est à l'origine sans rotation.

```python
ref = Referentiel()
ref.is_default()  # True

ref = Referentiel(x=1.0)
ref.is_default()  # False
```

---

## 2.3 PointCloud

Objet principal du projet. Contient les données géométriques, colorimétriques,
de classification et d'indexation d'un nuage de points.

### Attributs

| Attribut | Type | Obligatoire | Description |
|---|---|---|---|
| `xyz` | `np.ndarray (n, 3)` | ✅ | Coordonnées 3D |
| `rvb` | `np.ndarray (n, 3)` | ✅ | Couleurs RGB `uint8 [0-255]` |
| `classification` | `np.ndarray (n,)` | ❌ | Labels de classification |
| `indexation` | `np.ndarray (n,)` | ❌ | Indices des points |
| `info` | `PointCloudInfo` | ✅ | Métadonnées (créé automatiquement) |

### Méthodes de validation

#### `is_classified()`

Retourne `True` si le nuage possède une classification.

#### `is_indexed()`

Retourne `True` si le nuage possède une indexation.

#### `is_len_valid()`

Vérifie la cohérence des tailles entre `xyz`, `rvb`, `classification` et `indexation`.

```python
pc = PointCloud(xyz, rvb)
pc.is_len_valid()       # True

pc.classification = np.zeros(10)  # taille incorrecte
pc.is_len_valid()       # False
```

### Méthodes

#### `make_grid(axis="z")`

Calcule le `gridsize` du nuage pour la création d'une grille 2D.
Voir [section 9 — Rasterisation](../rasterisation) pour les détails.

```python
gridsize = pc.make_grid(axis="z")
# (x_min, x_max, y_min, y_max)
```

### Méthodes spéciales

| Méthode | Description |
|---|---|
| `__len__` | Retourne le nombre de points |
| `__repr__` | Affichage lisible en console |

```python
print(len(pc))   # 1_243_890
print(pc)        # PointCloud(n=1243890, classified=True, indexed=False)
```

---

## 2.4 Raster

Contient un raster 2D généré à partir d'un `PointCloud`, ainsi que toutes les
informations nécessaires à son géoréférencement et son export.

### Attributs

| Attribut | Type | Description |
|---|---|---|
| `raster` | `np.ndarray (H, W)` ou `(H, W, 3)` | Données raster |
| `resolution` | `float` | Taille d'un pixel en mètres |
| `gridsize` | `tuple` | `(x_min, x_max, y_min, y_max)` |
| `mode` | `str` | Mode de génération (`'m'`, `'a'`, `'c'`...) |
| `axis` | `str` | Axe de projection (`"z"`, `"-x"`...) |
| `offset` | `list[float]` | Offset du référentiel au moment de la génération |
| `angle` | `float` | Angle de rotation au moment de la génération (degrés) |
| `crs` | `str` | Système de coordonnées (ex: `"EPSG:2056"`) |

### Properties

| Property | Retour | Description |
|---|---|---|
| `shape` | `tuple` | Dimensions du raster |
| `width` | `int` | Nombre de colonnes |
| `height` | `int` | Nombre de lignes |
| `is_color` | `bool` | `True` si raster RGB (mode `'c'`) |

### Méthodes

#### `pixel_to_world(px, py)`

Convertit des coordonnées pixel en coordonnées monde, en tenant compte de la rotation.

```python
x, y = raster.pixel_to_world(100, 200)
```

#### `world_to_pixel(x, y)`

Convertit des coordonnées monde en coordonnées pixel, en tenant compte de la rotation.

```python
px, py = raster.world_to_pixel(2_600_000.5, 1_200_000.3)
```

#### `export(path)`

Exporte le raster en image et génère automatiquement le world file associé.
Voir [section 5 — Import / Export](../import_export) pour les formats supportés.

```python
raster.export("output/scan.png")
# -> scan.png + scan.pgw
```

#### `write_tfw(path)`

Génère uniquement le world file de géoréférencement.

```python
raster.write_tfw("output/scan.png")
# -> scan.pgw
```

---

## Relations entre les objets

```
PointCloud
├── xyz, rvb, classification, indexation
└── info : PointCloudInfo
        ├── name, source, date, crs
        ├── offset, angle  (référentiel local)
        └── history

Referentiel
└── x, y, z, a
    utilisé par refGlob2refLoc / refLoc2refGlob
    pour transformer un PointCloud

Raster
└── généré par pc_rasterise(PointCloud)
    exporté via raster_to_image() ou raster.export()
```
