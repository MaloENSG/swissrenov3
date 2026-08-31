# Rasterisation

> Conversion d'un nuage de points 3D en image raster 2D par projection selon un axe.

---

## Présentation générale

La rasterisation transforme un **nuage de points** (`PointCloud`) en une image 2D en projetant
les points sur un plan défini par un axe (`x`, `y` ou `z`). Le résultat est une image et ses métadonnées stocké dans la
table `raster_pcd` et associé au nuage source.

Cette outil permet de :

- générer des vues en plan (projection `z`) ou en coupe (projections `x` / `y`)
- servir de base pour la détection automatique avec des modèles d'IA
- réaliser des ortho images pour la digitalisation manuelle
- créer des illustrations pour communiquer sur des problématiques particulières

---

## Utilisation de l'outil

> **TODO** : Décrire comment lancer l'outil dans le POTREE.


---

## Paramètres

| Paramètre | Type | Valeurs | Description |
|---|---|---|---|
| `axis` | `String` | `"x"` \| `"y"` \| `"z"`\| `"-x"` \| `"-y"`\| `"-z"` | Axe de projection suivant les 3 axes dans les deux sens |
| `resolution` | `Float` | - | Taille d'un pixel en mètres |
| `angle` | `Float` | - | Rotation appliquée au nuage avant projection (degrés) |
| `mode` | `String` | `"accumulation"` \| `"hauteur max"` \| `"couleur"` | Mode de calcul de la valeur par pixel (**TODO** : valeurs possibles ?) |
| `is_aligned` | `Boolean` | ❌ | Indique si le raster a été réaligné après génération |

**Champs calculés automatiquement**

| Champ | Description |
|---|---|
| `gridsize` | Emprise de la grille `[xmin, xmax, ymin, ymax]` dans le référentiel du nuage de points |
| `center` | Centre de la grille `{"x": float, "y": float}` |
| `offset` | Décalage de coordonnées `(x, y, z)` hérité du nuage source ? |

**Paramètre `mode`**

| Mode | ID | Description |
|---|---|---|
| Accumulation | `a` | Accumulation ou densité par pixel. Correspond au nombre de points dans le pixel |
| Hauteur max | `m` | Hauteur du point le plus proche dans le pixel |
| Hauteur min | `n` | Hauteur du point le plus loin dans le pixel |
| Couleur | `c` | Couleur du points le plus proche |
| Binarisation | `p` | Présence ou non de points dans le pixel |
| Rugosité | `e` | Indice de rugosité, 0=lisse - 1=rugueux |

---

## TODO récapitulatif

- [ ] Contexte métier de la rasterisation
- [ ] Commande / interface pour lancer l'outil
- [ ] Valeurs possibles de `mode`
- [ ] Exemples de résultats (projections X, Y, Z)
- [ ] Limitations connues (densité minimale, cas limites…)
- [ ] Pipeline complet (place dans le flux global)
- [ ] Performances & recommandations (résolution selon usage)
- [ ] Formats de sortie (GeoTIFF, PNG…)