# CRUD — Opérations sur la base de données

Le module `bdd/crud.py` expose des fonctions d'insertion pour chaque entité de la BDD.
Toutes les fonctions suivent le même pattern : elles reçoivent une `session` SQLAlchemy,
valident les données, insèrent l'objet et retournent l'instance créée.

## Conventions

### Pattern commun

Chaque fonction `add_*` suit ce schéma :

```python
with Session(engine) as session:
    obj = crud.add_xxx(session, ...)
    session.commit()
```

!!! warning
    Le `session.commit()` n'est **pas** fait par les fonctions CRUD.
    C'est à l'appelant de committer ou rollback selon le résultat.

### Gestion des erreurs

Deux helpers internes gèrent les cas d'erreur :

| Helper | Rôle |
|--------|------|
| `_check_fk(session, model, pk, label)` | Vérifie qu'une clé étrangère existe, lève `ValueError` sinon |
| `_flush(session, obj, label)` | Ajoute et flush, rollback propre si `IntegrityError` |

Toutes les fonctions lèvent `ValueError` en cas de données invalides ou de contrainte violée.

### Géométries

Les géométries Shapely sont converties en WKB via `from_shape(..., srid=srid_swissrenov)`.
Le SRID utilisé est **2056** (Swiss LV95) pour les géométries 3D, **0** pour les géométries 2D panoramiques.

Les champs encodés en binaire (listes, matrices) utilisent `_encode()` / `_decode()` définis dans `models.py`.

---

## Entités du bâtiment

### `add_site`

```python
add_site(session, nom, centroide=None, extent_poly=None) → Site
```

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `nom` | `str` | ✓ | Nom unique du site |
| `centroide` | `Point` | — | Point Shapely (x, y) |
| `extent_poly` | `Polygon` | — | Emprise du site |

**Erreurs** : `ValueError` si le nom existe déjà.

---

### `add_parcelle`

```python
add_parcelle(session, numero, commune, egrid, nom_site=None,
             surface=None, extent_poly=None) → Parcelle
```

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `numero` | `str` | ✓ | Numéro de parcelle |
| `commune` | `str` | ✓ | Commune |
| `egrid` | `str` | ✓ | Identifiant EGRID (clé primaire) |
| `nom_site` | `str` | — | FK → `Site.nom` |
| `surface` | `float` | — | Surface en m² |
| `extent_poly` | `Polygon` | — | Emprise |

**Erreurs** : `ValueError` si `nom_site` introuvable ou `egrid` déjà existant.

---

### `add_batiment`

```python
add_batiment(session, egid, parcelle_egrid=None, nom_site=None,
             angle=None, offset=None, extent_poly=None) → Batiment
```

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `egid` | `int` | ✓ | Identifiant EGID (clé primaire) |
| `parcelle_egrid` | `str` | — | FK → `Parcelle.egrid` |
| `nom_site` | `str` | — | FK → `Site.nom` |
| `angle` | `float` | — | Angle d'orientation |
| `offset` | `tuple[float,float,float]` | — | Décalage (x, y, z) |
| `extent_poly` | `Polygon` | — | Emprise |

**Erreurs** : `ValueError` si `parcelle_egrid` ou `nom_site` introuvables, ou `egid` déjà existant.

---

### `add_volume`

```python
add_volume(session, egid_batis, nom=None, modelisation=None,
           volume=None, surface=None, extent_poly=None) → Volume
```

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `egid_batis` | `int` | ✓ | FK → `Batiment.egid` |
| `nom` | `str` | — | Nom du volume |
| `modelisation` | `str` | — | Type de modélisation |
| `volume` | `float` | — | Volume en m³ |
| `surface` | `float` | — | Surface en m² |

---

### `add_niveau`

```python
add_niveau(session, id_volume, nom=None, hmin=None, hmax=None,
           volume=None, surface=None) → Niveau
```

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `id_volume` | `int` | ✓ | FK → `Volume.id` |
| `nom` | `str` | — | Nom du niveau (ex: RDC, R+1) |
| `hmin` | `float` | — | Hauteur minimale en m |
| `hmax` | `float` | — | Hauteur maximale en m |

---

### `add_room`

```python
add_room(session, id_niveau, nom=None, volume=None,
         surface=None, extent_poly=None) → Room
```

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `id_niveau` | `int` | ✓ | FK → `Niveau.id` |
| `nom` | `str` | — | Nom de la pièce |
| `extent_poly` | `Polygon` | — | Emprise de la pièce |

---

## Panoramiques

### `add_panoramique`

```python
add_panoramique(session, sequencename, panoname,
                date_acquisition=None, date_traitement=None,
                camera_model=None, resolution=None,
                position=None, orientation=None) → Panoramique
```

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `sequencename` | `str` | ✓ | Nom de la séquence |
| `panoname` | `str` | ✓ | Nom du panoramique (ex: `pano_0075`) |
| `position` | `Point` | — | Position 3D (x, y, z) en LV95 |
| `orientation` | `list[list[float]]` | — | Matrice de rotation 3×3 |

**Erreurs** : `ValueError` si `orientation` n'est pas une matrice 3×3.

#### Import en masse

```python
crud.import_sequence_json(session, "chemin/vers/sequence.json")
```

Importe tous les panoramiques depuis un fichier JSON structuré :

```json
{
  "sequence_info": {
    "sequencename": "GS018175",
    "date_acquisition": "2025-07-08",
    "camera": "GoPro Max"
  },
  "pano_list": [
    {
      "panoname": "pano_0075",
      "position": [2538123.0, 1181456.0, 490.5],
      "orientation": [[1,0,0],[0,1,0],[0,0,1]]
    }
  ]
}
```

---

## Annotations

### `add_geometry_2d`

```python
add_geometry_2d(session, id_pano, type_geometry,
                panoname=None, sequencename=None,
                geometry_geom=None, segment_index=None,
                centroide_2d=None) → Geometry2D
```

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `id_pano` | `int` | ✓ | FK → `Panoramique.id` |
| `type_geometry` | `TypeGeometry` | ✓ | `POINT`, `LINE`, `POLYGON` ou `SEGMENT` |
| `geometry_geom` | `Point\|LineString\|Polygon` | Selon type | Géométrie Shapely (pas pour SEGMENT) |
| `segment_index` | `list[int]` | Pour SEGMENT | Liste d'indices SAM |
| `centroide_2d` | `Point` | — | Centroïde en coordonnées image |

**Règles de cohérence :**

| `type_geometry` | `geometry_geom` | `segment_index` |
|-----------------|-----------------|-----------------|
| `POINT` | requis | interdit |
| `LINE` | requis | interdit |
| `POLYGON` | requis | interdit |
| `SEGMENT` | interdit | requis |

---

### `add_annotation`

```python
add_annotation(session, id_pano=None, panoname=None,
               sequencename=None, label=None, commentaire=None,
               permission=None, date_saisie=None, auteur=None,
               domaine_form=None, id_union_geom=None,
               centroide_3d=None) → Annotation
```

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `id_pano` | `int` | ✓ | FK → `Panoramique.id` |
| `panoname` | `str` | — | Nom du panoramique (dénormalisé) |
| `sequencename` | `str` | — | Nom de la séquence (dénormalisé) |
| `domaine_form` | `str` | — | FK → `Formulaire.name` |
| `id_union_geom` | `list[int]` | — | Liste de FK → `Geometry2D.id` |
| `centroide_3d` | `Point` | — | Position 3D en LV95 |

**Erreurs** : `ValueError` si `domaine_form` introuvable ou si un id dans `id_union_geom` est inexistant.

!!! note
    `id_union_geom` est une liste car une annotation peut référencer plusieurs géométries (ex: multi-polygone SAM).

---

## Formulaires

### `add_formulaire`

```python
add_formulaire(session, name, description=None) → Formulaire
```

**Erreurs** : `ValueError` si le `name` existe déjà.

---

### `add_formulaire_champ`

```python
add_formulaire_champ(session, id_formulaire, nom, type_champ,
                     ordre=0, obligatoire=False, description=None,
                     options=None, source_table=None,
                     source_label_col=None) → FormulaireChamp
```

| `type_champ` | Paramètres requis |
|--------------|-------------------|
| `TEXT` | — |
| `NUMBER` | — |
| `DATE` | — |
| `SELECT` | `options: list[str]` |
| `SELECT_TABLE` | `source_table`, `source_label_col` |

**Erreurs** : `ValueError` si incohérence entre `type_champ` et les paramètres fournis.

---

### `add_formulaire_reponse`

```python
add_formulaire_reponse(session, id_formulaire, id_annotation=None,
                       auteur=None, date_saisie=None,
                       valeurs=None) → FormulaireReponse
```

| Paramètre | Type | Description |
|-----------|------|-------------|
| `id_formulaire` | `int` | FK → `Formulaire.id` |
| `id_annotation` | `int` | FK → `Annotation.id` |
| `valeurs` | `dict` | `{"nom_champ": valeur, ...}` |

**Validation des valeurs :**

- Les clés de `valeurs` doivent correspondre aux champs du formulaire
- Les champs `obligatoire=True` doivent être présents dans `valeurs`

**Erreurs** : `ValueError` si clés inconnues ou champs obligatoires manquants.

---

## Nuage de points

### `add_pointcloud`

```python
add_pointcloud(session, egid_batis=None, nom=None, ...) → PointCloud
```

Paramètres notables :

| Paramètre | Type | Description |
|-----------|------|-------------|
| `scan_position` | `Point` | Position 3D du scanner |
| `scan_rotation` | `list[list[float]]` | Matrice de rotation 3×3 |
| `bbox` | `list[tuple]` | Boîte englobante (liste de points 3D) |
| `offset` | `tuple[float,float,float]` | Décalage (x, y, z) |
| `is_merged` | `bool` | Nuage fusionné |
| `is_classified` | `bool` | Nuage classifié |

---

### `add_map3d`

```python
add_map3d(session, id_pcd=None, sequencename=None,
          foldername=None) → Map3D
```

---

### `add_raster_pcd`

```python
add_raster_pcd(session, id_pcd=None, name=None, resolution=None,
               mode=None, axis=None, angle=None, is_aligned=False,
               gridsize=None, center=None, offset=None) → RasterPcd
```

| Paramètre | Contrainte |
|-----------|-----------|
| `axis` | Doit être `'x'`, `'y'` ou `'z'` |
| `gridsize` | Liste de 4 valeurs `[xmin, xmax, ymin, ymax]` |

---

## Maquette

### `add_maquette_gbxml`

```python
add_maquette_gbxml(session, egid_batis, date_generation=None,
                   attribut1=None, attribut2=None,
                   regbl_path=None, offset=None) → MaquetteGbxml
```