# Architecture de la base de données

> Base de données pour le viewer d'images panoramiques Swissrenov avec gestion des annotations et de la connexion à la 3D.

---

## Vue d'ensemble

La base de données est organisée autour de trois grands axes qui se rejoignent au niveau du **Bâtiment** :

- **Axe géographique** : `Site` → `Parcelle` → `Bâtiment` → `Volume` → `Niveau` → `Room`
- **Axe d'annotation** : `Panoramique` → `Geometry2D` → `Annotation` → `Formulaire`
- **Axe nuage de points** : `PointCloud` → `RasterPcd`/`Map3D`

Les formulaires (`Formulaire`, `FormulaireChamp`, `FormulaireReponse`) permettent d'attacher des données structurées et personnalisables à chaque annotation.
Les maquettes `MaquetteGbxml` sont des données supplémentaires rattachées au niveau du **Bâtiment**.

Le système de coordonnées de référence est le **SRID 2056** (MN95 / LV95 — réseau suisse).

---

## Diagramme des relations

```
Site
 ├── Parcelle (N)
 │    └── Bâtiment (N)
 │         ├── Volume (N)
 │         │    └── Niveau (N)
 │         │         └── Room (N)
 │         ├── PointCloud (N)
 │         │    ├── RasterPcd (N)
 │         │    └── Map3D (N)
 │         └── MaquetteGbxml (N)
 │
Panoramique
 ├── Geometry2D (N)
 └── Annotation (N)
      ├── FormulaireReponse (N)
      └── → Formulaire
               └── FormulaireChamp (N)

Auteur
 ├── → Annotation (N)
 └── → FormulaireReponse (N)

MateriauxTypes   (table de référence autonome utilisable par un FormulaireChamp)
```

> **TODO** : Générer et insérer un diagramme ERD (ex. avec `eralchemy`, `dbdiagram.io` ou `mermaid`) pour remplacer ou compléter le schéma texte ci-dessus.

---

## Tables

### `site`

Représente une friche (ensemble de parcelles et de bâtiments).

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `nom` | `String(256)` | **PK** | Identifiant unique du site |
| `centroide` | `POINT` (SRID 2056) | nullable | Centre géographique du site |
| `extent_poly` | `POLYGON` (SRID 2056) | nullable | Emprise géographique du site |

**Relations**

- `parcelles` → `Parcelle` (1-N)
- `batiments` → `Batiment` (1-N)

!!! info "Informations sur la friche"
    Cette table réunit les informations générales sur une friche. Des champs attributaires peuvent
    être ajouté pour compléter la base de données. De nombreuses informations sont disponibles via
    l'API swisstopo (pollution, ensoleillement, géométrie, RegBL, desserte TP ... etc).


---

### `parcelle`

Représente une parcelle du cadastre.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `egrid` | `String(128)` | **PK** | Identifiant fédéral de parcelle (EGRID) |
| `numero` | `String(64)` | NOT NULL | Numéro de parcelle |
| `commune` | `String(128)` | NOT NULL | Commune de la parcelle |
| `surface` | `Float` | nullable | Surface en m² |
| `nom_site` | `String(256)` | FK → `site.nom`, nullable | Site de rattachement |
| `extent_poly` | `POLYGON` (SRID 2056) | nullable | Emprise géographique |

**Relations**

- `site` → `Site` (N-1)
- `batiments` → `Batiment` (1-N)

---

### `batiment`

Représente un bâtiment au sens du registre fédéral des bâtiments et logements (RegBL).

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `egid` | `Integer` | **PK** (non auto) | Identifiant fédéral du bâtiment (EGID) |
| `angle` | `Float` | nullable | Angle d'orientation du bâtiment par rapport au nord (degrés sens anti-horaire) |
| `parcelle_egrid` | `String` | FK → `parcelle.egrid`, nullable | Parcelle de rattachement |
| `nom_site` | `String` | FK → `site.nom`, nullable | Site de rattachement direct |
| `offset` | `JSON → (x, y, z)` | nullable | Décalage de coordonnées pour la définition de l'origine local |
| `extent_poly` | `POLYGON` (SRID 2056) | nullable | Emprise au sol |

**Relations**

- `parcelle` → `Parcelle` (N-1)
- `site` → `Site` (N-1)
- `volumes` → `Volume` (1-N)
- `pointclouds` → `PointCloud` (1-N)
- `maquettes_gbxml` → `MaquetteGbxml` (1-N)

!!! info "Paramètres du bâtiment"
    Les attributs `angle` et `offset` sont des paramètres utilent pour centrer et aligner le bâtiment dans un référentiel local
    tout en conservant le géoréférencement.

---

!!! warning "Construction des sous-divisions `volume`-`niveau`-`room`"
    Il n'y a pas d'interface permettant une construction des `volume`, `niveau` et `room` pour le moment. Ce sont des éléments de
    segmentation du bâtiment pour aider l'utilisateur à organiser et traiter les données et informations. En fonction de l'état
    d'avancement du chantier, il est possible de réaliser plusieurs "construction", l'attribut `modelisation` permet de les nommer
    pour le distinguer.

    Cette hiérarchie n'est pas nécessaire pour le bon fonctionnement du projet.

### `volume`

Sous-division volumétrique d'un bâtiment (ex. : cage d'escalier, aile, bloc…).

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant |
| `nom` | `String(256)` | nullable | Nom du volume |
| `modelisation` | `String(256)` | nullable | Nom de la modélisation |
| `volume` | `Float` | nullable | Volume en m³ |
| `surface` | `Float` | nullable | Surface au sol en m² |
| `egid_batis` | `Integer` | FK → `batiment.egid`, nullable | Bâtiment parent |
| `extent_poly` | `POLYGON` (SRID 2056) | nullable | Emprise au sol |

**Relations**

- `batiment` → `Batiment` (N-1)
- `niveaux` → `Niveau` (1-N)

---

### `niveau`

Étage ou niveau d'un volume.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `nom` | `String(256)` | nullable | Nom ou label du niveau (ex. : « RDC », « R+1 ») |
| `hmin` | `Float` | nullable | Hauteur minimale |
| `hmax` | `Float` | nullable | Hauteur maximale |
| `volume` | `Float` | nullable | Volume du niveau en m³ |
| `surface` | `Float` | nullable | Surface du niveau en m² |
| `id_volume` | `Integer` | FK → `volume.id`, nullable | Volume parent |

**Relations**

- `volume_obj` → `Volume` (N-1)
- `rooms` → `Room` (1-N)

!!! info "Paramètres du niveau"
    La hauteur mininale et maximale d'un niveau est exprimé en mètre dans le référentiel du nuage de point

---

### `room`

Pièce ou local au sein d'un niveau.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `nom` | `String(256)` | nullable | Nom de la pièce |
| `volume` | `Float` | nullable | Volume en m³ |
| `surface` | `Float` | nullable | Surface en m² |
| `id_niveau` | `Integer` | FK → `niveau.id`, nullable | Niveau parent |
| `extent_poly` | `POLYGON` (SRID 2056) | nullable | Emprise au sol |

**Relations**

- `niveau` → `Niveau` (N-1)

---

### `maquette_gbxml`

Maquette numérique d'un bâtiment au format gbXML.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant |
| `date_generation` | `String(32)` | nullable | Date de génération de la maquette |
| `attribut1` | `Float` | nullable | Exemple d'attribut |
| `attribut2` | `String(256)` | nullable | Exemple d'attribut |
| `regbl_path` | `String(512)` | nullable | Chemin vers le fichier `.gbxml` |
| `egid_batis` | `Integer` | FK → `batiment.egid`, nullable | EGID Bâtiment |
| `offset` | `JSON → (x, y, z)` | nullable | Origine de la maquette |

**Relations**

- `batiment` → `Batiment` (N-1)

!!! warning "Développement en cours"
    Les attributs `attribut1` et `attribut2` sont des **placeholders temporaires** issus du fichier
    Swissbuilding 3.0 utilisé pour générer la maquette gbXML. Les attributs définitifs seront
    intégrés à la table `maquette_gbxml` une fois leur structure stabilisée.

!!! info "Remarque"
    Pour plus d'informations sur les maquettes gbXML, se référer à la page TODO

---

### `pointcloud`

Nuage de points 3D (PCD) associé à un bâtiment.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant |
| `nom` | `String(256)` | nullable | Nom du PCD |
| `source_sensor` | `String(128)` | nullable | Capteur d'acquisition |
| `date_acquisition` | `String(32)` | nullable | Date d'acquisition |
| `date_traitement` | `String(32)` | nullable | Date de traitement |
| `crs` | `String(64)` | nullable | Système de coordonnées du PCD |
| `sampling` | `Integer` | nullable | Sous-échantillonnage (mm) |
| `angle` | `Float` | nullable | Angle d'orientation issue du `batiment` |
| `is_merged` | `Boolean` | défaut `False` | Nuage issue d'une fusion |
| `is_filtered` | `Boolean` | défaut `False` | Filtrage des outliers |
| `is_indexed` | `Boolean` | défaut `False` | Indexation des points du nuage |
| `is_classified` | `Boolean` | défaut `False` | Classification automatique |
| `egid_batis` | `Integer` | FK → `batiment.egid`, nullable | Bâtiment associé |
| `source_index` | `Integer` | FK → `pointcloud.id`, nullable | Provenance de l'indexation dans le cas ou le PCD est un sous ensemble d'un autre PCD |
| `source_idx_pcd` | `JSON → [int]` | nullable | Liste des ID(s) des PCD utilisé(s) pour constituer ce PCD (fusion, sous-ensemble) |
| `scan_position` | `JSON → {x, y, z}` | nullable | Position du scanner |
| `offset` | `JSON → (x, y, z)` | nullable | Décalage de coordonnées |
| `scan_rotation` | `JSON → 3×3` | nullable | Matrice de rotation du capteur (Sensor → World) |
| `bbox` | `JSON → [[x,y,z]×8]` | nullable | Bounding box 3D (8 sommets) |
| `extent_poly` | `POLYGON` (SRID 2056) | nullable | Emprise au sol |

**Relations**

- `batiment` → `Batiment` (N-1)
- `maps_3d` → `Map3D` (1-N)
- `rasters` → `RasterPcd` (1-N)

Lorsqu'un Nuage de points est crée et ajouté au projet, les métadonnées associé sont sauvegardé dans la table `pointcloud` pour garantir la traçabilité.
Cela permet d'abord de conserver les informations essentielles (capteur utilisé, date d'acquisition, CRS, échantillonnage ... etc), mais aussi les liens entre nuages de points.

Dans le cas ou un nuage de points provient d'un traitement sur un ou plusieurs nuage(s) de points source (fusion, sous-ensemble), le/les `id` de ces nuages de points sont stocké dans `source_idx_pcd`.
Si le nuage de points produit est un sous-ensemble d'un autre nuage qui possède une indexation, il est possible de conserver cette indexation avec `source_index`. Cela permet de conserver
la fusion 3D/images.


---

### `raster_pcd`

Image raster dérivée d'un nuage de points (projection selon un axe).

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant |
| `name` | `String(256)` | nullable | Nom du raster |
| `resolution` | `Float` | nullable | Résolution en m/pixel |
| `mode` | `String(128)` | nullable | Mode d'image |
| `axis` | `String(16)` | nullable | Axe de projection |
| `angle` | `Float` | nullable | Angle de rotation avant projection |
| `is_aligned` | `Boolean` | défaut `False` | Raster réaligné |
| `id_pcd` | `Integer` | FK → `pointcloud.id`, nullable | PCD source |
| `gridsize` | `JSON → [xmin, xmax, ymin, ymax]` | nullable | Emprise de la grille |
| `center` | `JSON → {x, y}` | nullable | Centre de la grille |
| `offset` | `JSON → (x, y, z)` | nullable | Décalage de coordonnées |

**Relations**

- `pointcloud` → `PointCloud` (N-1)

!!! info "Remarque"
    Pour plus d'informations sur les rasters, se référer à la page TODO

---

### `map_3d`

Carte ou scène 3D issue d'un nuage de points.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `sequencename` | `String(256)` | nullable | Nom de la séquence associée |
| `foldername` | `String(256)` | nullable | Dossier des données |
| `id_pcd` | `Integer` | FK → `pointcloud.id`, nullable | Nuage de points source |

**Relations**

- `pointcloud` → `PointCloud` (N-1)

> **TODO** : Préciser le format et l'usage de la Map3D (viewer web, export Potree, 3D Tiles…).

---

### `panoramique`

Image panoramique à 360° acquise sur le terrain.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `sequencename` | `String(256)` | NOT NULL | Nom de la séquence de capture |
| `panoname` | `String(256)` | NOT NULL | Nom de l'image panoramique |
| `date_acquisition` | `String(32)` | nullable | Date d'acquisition |
| `date_traitement` | `String(32)` | nullable | Date de traitement |
| `camera_model` | `String(128)` | nullable | Modèle de caméra |
| `resolution` | `String(64)` | nullable | Résolution de l'image |
| `position` | `POINTZ` (SRID 2056) | nullable | Position géographique 3D du centre optique |
| `orientation` | `JSON → 3×3` | nullable | Matrice de rotation de la caméra |

**Relations**

- `geometries_2d` → `Geometry2D` (1-N)
- `annotations` → `Annotation` (1-N)

> **TODO** : Décrire le mode d'acquisition (caméra embarquée, trépied, drone…) et le format de stockage des images.

---

### `geometry_2d`

Géométrie dessinée dans l'espace image d'un panoramique (coordonnées pixel).

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `panoname` | `String(256)` | nullable | Nom du panoramique (dénormalisé) |
| `sequencename` | `String(256)` | nullable | Nom de la séquence (dénormalisé) |
| `type_geometry` | `Enum(TypeGeometry)` | nullable | Type : `point`, `line`, `polygon`, `segment` |
| `id_pano` | `Integer` | FK → `panoramique.id`, nullable | Panoramique parent |
| `geometry_geom` | `GEOMETRY` (SRID 0) | nullable | Géométrie en coords pixel |
| `segment_index` | `Text` | nullable | Indice de segment (si type `segment`) |
| `centroide_2d` | `POINT` (SRID 0) | nullable | Centroïde en coords pixel |

**Enum `TypeGeometry`**

| Valeur | Description |
|---|---|
| `point` | Point unique |
| `line` | Polyligne |
| `polygon` | Polygone fermé |
| `segment` | Référence à un segment par indice entier |

**Relations**

- `panoramique` → `Panoramique` (N-1)

> **TODO** : Expliquer pourquoi `panoname` et `sequencename` sont dénormalisés (performance ? héritage ?).

---

### `annotation`

Observation pointée sur un panoramique, liée à une ou plusieurs géométries 2D et à un formulaire de saisie.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `panoname` | `String(256)` | nullable | Nom du panoramique (dénormalisé) |
| `sequencename` | `String(256)` | nullable | Nom de la séquence (dénormalisé) |
| `date_saisie` | `String(32)` | nullable | Date de saisie |
| `permission` | `Integer` | nullable | Niveau de permission (**TODO** : valeurs ?) |
| `label` | `String(256)` | nullable | Label / catégorie de l'annotation |
| `commentaire` | `Text` | nullable | Commentaire libre |
| `id_pano` | `Integer` | FK → `panoramique.id`, nullable | Panoramique associé |
| `auteur` | `Integer` | FK → `auteur.id`, nullable | Auteur de l'annotation |
| `domaine_form` | `String` | FK → `formulaire.name`, nullable | Formulaire de saisie associé |
| `id_union_geom` | `JSON → [int]` | nullable | Liste des IDs `Geometry2D` composant l'annotation |
| `centroide_3d` | `POINTZ` (SRID 2056) | nullable | Position 3D reconstruite de l'annotation |

**Relations**

- `panoramique` → `Panoramique` (N-1)
- `reponses` → `FormulaireReponse` (1-N)
- `auteur_obj` → `Auteur` (N-1)

> **TODO** : Décrire comment `centroide_3d` est calculé (raycasting sur nuage de points ?).  
> **TODO** : Clarifier la sémantique de `permission` (lecture seule, éditable, public…).

---

### `formulaire`

Définition d'un formulaire de saisie (structure, pas les données).

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant |
| `name` | `String(256)` | UNIQUE, NOT NULL | Identifiant textuel du formulaire |
| `description` | `Text` | nullable | Description du formulaire |

**Relations**

- `champs` → `FormulaireChamp` (1-N, ordonnés par `ordre`)
- `reponses` → `FormulaireReponse` (1-N)

---

### `formulaire_champ`

Un champ d'un formulaire (définit la structure, pas la valeur).

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `nom` | `String(256)` | NOT NULL | Clé du champ (utilisée dans le JSON des réponses) |
| `type_champ` | `Enum(TypeChamp)` | NOT NULL | Type de champ |
| `obligatoire` | `Boolean` | défaut `False` | Champ obligatoire |
| `ordre` | `Integer` | défaut `0` | Ordre d'affichage |
| `description` | `Text` | nullable | Description du champs |
| `options` | `JSON → [str]` | nullable | Options pour les types `select` |
| `id_formulaire` | `Integer` | FK → `formulaire.id`, CASCADE | Formulaire parent |
| `source_table` | `String(256)` | nullable | Table source pour `select_table` (ex. : `materiaux_types`) |
| `source_label_col` | `String(128)` | nullable | Colonne à afficher pour `select_table` (ex. : `nom`) |

**Enum `TypeChamp`**

| Valeur | Description |
|---|---|
| `text` | Champ texte court |
| `textarea` | Champ texte long |
| `number` | Valeur numérique |
| `boolean` | Oui / Non |
| `select` | Liste déroulante (options statiques dans `_options`) |
| `date` | Date |
| `select_table` | Liste déroulante dynamique depuis une table comme materiaux_types |

**Relations**

- `formulaire` → `Formulaire` (N-1)

---

### `formulaire_reponse`

Une réponse complète à un formulaire, liée à une annotation.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `date_saisie` | `String(32)` | nullable | Date de saisie de la réponse |
| `id_formulaire` | `Integer` | FK → `formulaire.id`, CASCADE | Formulaire concerné |
| `id_annotation` | `Integer` | FK → `annotation.id`, nullable | Annotation associée |
| `auteur` | `Integer` | FK → `auteur.id`, nullable | Auteur de la réponse |
| `valeurs` | `JSON → {clé: valeur}` | nullable | Valeurs saisies (clés = `nom` des champs) |

**Exemple de `valeurs`**

```json
{
  "gravite": "moyen",
  "surface_m2": 12.5,
  "commentaire": "fissure en façade"
}
```

**Relations**

- `formulaire` → `Formulaire` (N-1)
- `annotation` → `Annotation` (N-1)
- `auteur_obj` → `Auteur` (N-1)

---

### `auteur`

Utilisateur ou opérateur ayant saisi des données.

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `nom` | `String(128)` | NOT NULL | Nom de famille |
| `prenom` | `String(128)` | nullable | Prénom |
| `date_enregistrement` | `String(32)` | nullable | Date de création du compte |
| `entreprise` | `String(256)` | nullable | Entreprise / organisation |
| `role` | `String(128)` | nullable | Rôle (**TODO** : valeurs possibles ?) |

**Relations**

- `annotations` → `Annotation` (1-N)
- `reponses` → `FormulaireReponse` (1-N)

> **TODO** : Préciser si `auteur` est lié à un système d'authentification externe (SSO, OAuth…).  
> **TODO** : Documenter les valeurs possibles de `role`.

---

### `materiaux_types`

Table de référence des types de matériaux (utilisée comme source pour des champs `select_table`).

| Colonne | Type | Contrainte | Description |
|---|---|---|---|
| `id` | `Integer` | **PK** auto | Identifiant interne |
| `nom` | `String(256)` | UNIQUE, NOT NULL | Nom du matériau |
| `description` | `Text` | nullable | Description du matériau |

> **TODO** : Lister les matériaux de référence initialement peuplés dans la table.  
> **TODO** : Décrire le processus de maintenance de cette table (qui peut ajouter / modifier des entrées ?).

---

## Conventions techniques

### Champs géographiques

Toutes les géométries géoréférencées utilisent le **SRID 2056** (MN95 / LV95). Les géométries 2D en espace image (`Geometry2D`) utilisent le **SRID 0** (coordonnées pixel sans projection).

La bibliothèque utilisée est [GeoAlchemy2](https://geoalchemy-2.readthedocs.io/).

### Champs JSON

Certaines colonnes stockent des structures complexes sérialisées en JSON dans des champs `Text`. Elles sont exposées via des propriétés Python (`@property`) avec validation à la saisie :

| Propriété | Format attendu | Tables concernées |
|---|---|---|
| `offset` | `(x, y, z)` | `Batiment`, `PointCloud`, `MaquetteGbxml`, `RasterPcd` |
| `scan_position` | `{"x": float, "y": float, "z": float}` | `PointCloud` |
| `scan_rotation` | matrice 3×3 | `PointCloud` |
| `orientation` | matrice 3×3 | `Panoramique` |
| `bbox` | liste de points `[x, y, z]` | `PointCloud` |
| `gridsize` | `[xmin, xmax, ymin, ymax]` | `RasterPcd` |
| `center` | `{"x": float, "y": float}` | `RasterPcd` |
| `valeurs` | `dict` libre | `FormulaireReponse` |
| `options` | `[str]` | `FormulaireChamp` |
| `id_union_geom` | `[int]` | `Annotation` |
| `source_idx_pcd` | `[int]` | `PointCloud` |

### Comportement des clés étrangères

| Comportement | Usage |
|---|---|
| `ON DELETE SET NULL` | Relation facultative — l'enregistrement fils survit à la suppression du parent |
| `ON DELETE CASCADE` | Relation obligatoire — la suppression du parent entraîne celle des fils (ex. : `FormulaireChamp`, `FormulaireReponse`) |

### Dates

Les dates sont stockées en `String(32)` sous le format `YYYY-MM-DD`.

---

## TODO récapitulatif

- [ ] Diagramme ERD visuel
- [ ] Clarifier la redondance `nom_site` sur `Batiment` vs via `Parcelle`
- [ ] Référentiel altimétrique pour `hmin` / `hmax` dans `Niveau`
- [ ] Format et stockage physique des fichiers PointCloud
- [ ] Valeurs possibles de `mode` dans `RasterPcd`
- [ ] Usage de `Map3D` (Potree, 3D Tiles…)
- [ ] Mode d'acquisition et stockage des images `Panoramique`
- [ ] Explication de la dénormalisation `panoname` / `sequencename` dans `Geometry2D` et `Annotation`
- [ ] Sémantique de `permission` dans `Annotation`
- [ ] Calcul de `centroide_3d` dans `Annotation`
- [ ] Liste des formulaires existants en production
- [ ] Valeurs possibles de `role` dans `Auteur`
- [ ] Lien éventuel entre `Auteur` et un système d'authentification
- [ ] Contenu initial de `MateriauxTypes`
