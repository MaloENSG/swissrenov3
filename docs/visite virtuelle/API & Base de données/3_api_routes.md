# API Backend — Référence des routes

Le serveur Flask tourne sur `http://127.0.0.1:5000`. Toutes les routes sont préfixées par `/api/`.
CORS activé pour toutes les origines.

---

## Panoramiques

### `POST /api/markers`

Retourne la liste des panoramiques dans une emprise spatiale.

**Corps de la requête :**
```json
{
  "seqname": "GS018175",
  "xmin": 2538000, "xmax": 2539000,
  "ymin": 1181000, "ymax": 1182000,
  "zmin": 480,     "zmax": 510
}
```

!!! note
    `seqname: "all"` retourne toutes les séquences sans filtre.

**Réponse `200` :**
```json
[
  {
    "id": 1,
    "nom": "pano_0075",
    "sequencename": "GS018175",
    "x_coord": 2538123.456,
    "y_coord": 1181456.789,
    "z_coord": 490.5,
    "date": "2025-07-08"
  }
]
```

---

### `POST /api/seq`

Retourne la liste des séquences disponibles.

**Corps de la requête :** `{}` *(corps ignoré)*

**Réponse `200` :**
```json
[
  { "id": 0, "sequencename": "all" },
  { "id": 1, "sequencename": "GS018175" },
  { "id": 2, "sequencename": "GS018176" }
]
```

---

## Mesure

### `POST /api/coordpts`

Retourne les coordonnées 3D (LV95) d'un point cliqué dans le panorama.

**Corps de la requête :**
```json
{
  "panoname": "0075",
  "seqname": "GS018175",
  "param": [[1.234, -0.456]]
}
```

`param[0]` est `[yaw, pitch]` en radians.

**Réponse `200` :**
```json
{ "message": "X : 2538123.456<br>Y : 1181456.789<br>Z : 490.500" }
```

---

### `POST /api/distpano`

Retourne les coordonnées 3D de plusieurs points pour calculer des distances.

**Corps de la requête :**
```json
{
  "panoname": "0075",
  "seqname": "GS018175",
  "param": [
    [1.234, -0.456],
    [1.567, -0.123]
  ]
}
```

**Réponse `200` :**
```json
{
  "X": [12.345, 15.678],
  "Y": [5.432, 6.789],
  "Z": [1.234, 1.456]
}
```

Les coordonnées sont en repère local (sans offset géographique).

---

## Segmentation SAM

### `POST /api/extractSAM`

Retourne le(s) contour(s) du segment SAM cliqué dans le panorama.

**Corps de la requête :**
```json
{
  "panoname": "0075",
  "seqname": "GS018175",
  "param": [[1.234, -0.456]]
}
```

**Réponse `200` — liste de polygones :**
```json
[
  {
    "poly_0": {
      "panoname": "0075",
      "seqname": "GS018175",
      "indiceSAM": 42,
      "poly": [[120, 340], [121, 341], ...]
    }
  }
]
```

`poly` est soit une liste de points `[[x,y],...]` (polygone simple), soit une liste de listes `[[[x,y],...], [[x,y],...]]` (polygone avec trous).

Retourne `[]` si le clic ne touche aucun segment.

---

## Annotations

### `POST /api/save_annotation`

Crée une géométrie 2D et une annotation associée, avec optionnellement une réponse formulaire.

**Corps de la requête :**
```json
{
  "panoname": "0075",
  "seqname": "GS018175",
  "param": {
    "geometry_type": "Polygon",
    "uv_coord": [[1.2, -0.4], [1.3, -0.5], [1.1, -0.5]],
    "segment_id": null,
    "domaine_form": "test1",
    "valeurs": { "nom": "caisse", "surface": 1.5 },
    "label": null,
    "commentaire": null,
    "permission": null
  }
}
```

!!! note
    `panoname` est automatiquement préfixé par `pano_` côté serveur.

**Règles selon `geometry_type` :**

| `geometry_type` | `uv_coord` | `segment_id` |
|-----------------|------------|--------------|
| `Point` | `[yaw, pitch]` | — |
| `Line` | `[[yaw,pitch], ...]` | — |
| `Polygon` | `[[yaw,pitch], ...]` | — |
| `Segment` | — | `[42, 57, ...]` (indices SAM) |

**Réponse `201` :**
```json
{
  "status": "ok",
  "annotation_id": 12,
  "geometry_id": 8
}
```

**Erreurs :**

| Code | Cause |
|------|-------|
| `404` | Panoramique introuvable |
| `400` | `segment_id` manquant pour type `Segment` |
| `400` | Erreur de validation CRUD |

---

### `POST /api/affichage_objet`

Route en 3 étapes selon les paramètres fournis.

**Étape 1 — Liste des formulaires disponibles sur le pano :**
```json
{ "panoname": "0075", "seqname": "GS018175", "param": {} }
```
```json
{ "step": "choose_formulaire", "formulaires": ["test1", "materiaux"] }
```

**Étape 2 — Liste des champs du formulaire :**
```json
{ "panoname": "0075", "seqname": "GS018175", "param": { "formulaire": "test1" } }
```
```json
{
  "step": "choose_champ",
  "champs": [
    { "nom": "nom", "type_champ": "text" },
    { "nom": "surface", "type_champ": "number" }
  ]
}
```

**Étape 3 — Résultats filtrés :**
```json
{
  "panoname": "0075",
  "seqname": "GS018175",
  "param": {
    "formulaire": "test1",
    "champ": "nom",
    "valeur": "caisse"
  }
}
```
```json
{
  "step": "results",
  "formulaire": "test1",
  "champ": "nom",
  "valeur": "caisse",
  "count": 2,
  "annotations": [
    {
      "annotation_id": 6,
      "label": null,
      "commentaire": null,
      "nom": "caisse",
      "geometries": [
        {
          "id": 4,
          "type_geometry": "polygon",
          "segment_index": [[1.2, -0.4], [1.3, -0.5], ...]
        }
      ]
    }
  ]
}
```

!!! note
    Si `valeur` est omis, toutes les annotations du champ sont retournées.
    Pour `type_geometry: "segment"`, `segment_index` contient une liste de polygones PSV-compatibles calculés depuis les indices SAM.

---

### `DELETE /api/delete_annotation/<annotation_id>`

Supprime une annotation et toutes ses données associées (géométries, réponses formulaire).

**Exemple :** `DELETE /api/delete_annotation/12`

**Réponse `200` :**
```json
{ "status": "ok", "deleted": 12 }
```

**Erreur `404` :** annotation introuvable.

!!! warning
    Suppression définitive et en cascade : `Geometry2D` et `FormulaireReponse` liés sont supprimés.

---

## Formulaires

### `GET /api/formulaires`

Retourne tous les formulaires avec leurs champs.

**Réponse `200` :**
```json
[
  {
    "id": 1,
    "name": "test1",
    "description": "Formulaire de test",
    "champs": [
      {
        "nom": "nom",
        "type_champ": "text",
        "obligatoire": true,
        "ordre": 0,
        "description": null,
        "options": null,
        "source_table": null,
        "source_label_col": null
      }
    ]
  }
]
```

---

### `POST /api/formulaire`

Crée un formulaire avec ses champs en une seule requête.

**Corps de la requête :**
```json
{
  "name": "materiaux",
  "description": "Identification des matériaux",
  "champs": [
    {
      "nom": "type",
      "type_champ": "select",
      "obligatoire": true,
      "ordre": 0,
      "options": ["béton", "bois", "métal", "verre"]
    },
    {
      "nom": "etat",
      "type_champ": "text",
      "obligatoire": false,
      "ordre": 1
    }
  ]
}
```

**Réponse `201` :**
```json
{
  "status": "ok",
  "formulaire_id": 3,
  "formulaire_name": "materiaux",
  "nb_champs": 2
}
```

**Erreurs :**

| Code | Cause |
|------|-------|
| `400` | `name` manquant ou aucun champ |
| `409` | Formulaire avec ce nom existe déjà |
| `400` | Erreur de validation sur un champ |
