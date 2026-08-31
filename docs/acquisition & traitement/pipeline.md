# Pipeline & Récapitulatif

> Transforme une séquence vidéo en nuages de points 3D nettoyés et en panoramiques
> géoréférencés prêts pour les viewers Swissrenov.


Faire tableau recap avec temps de traitement + vitesse de traitement. Faire aussi stockage volume. Faire schéma récap

---

## Vue d'ensemble

Le pipeline se divise en deux branches parallèles alimentées par la même vidéo source,
qui convergent vers des étapes optionnelles de post-traitement.

- **Branche PCD** (étapes 1–3) — produit un nuage de points nettoyé (`PointCloud`)
- **Branche Panoramiques** (étapes 4–6) — produit des panoramiques tuilés (`Panoramique`)
- **Post-traitement optionnel** (étapes 7–8) — reprojection et segmentation automatique

---

## Étapes

### 1. Extraction de frames (MODULE TODO PAGE ??)

**Entrée** : vidéo source au format `video-name.360` issue d'une caméra panoramique (GoPro max 2). D'autres modéles de caméra sont utilisables mais cela nécéssite une adaptation du module.

**Sortie** : images individuelles `.png` nommé suivant le format `video-name_caméra_numéro` (exemple GS0200_Bck_0060) stocké dans un dossier `video-name`

**Durée estimée** : TODO

Extrait les images de la vidéo qui serviront au calcul photogrammétrique avec le MODULE TODO. La fréquence d'extraction conseillé est de 2 images par second.

---

### 2. Calcul photogrammétrique

**Entrée** : Séquence de frames.

**Sortie** : Nuage de points brut `.las`, poses caméra `OPK`. (optionnel : points de calages pour le géoréférencement)

**Durée estimée** : TODO (variable selon le nombre de frames et la puissance de calcul)

Reconstruction photogrammétrique et géoréférencement : Cette étape est la seule étape du processus à nécéssiter un travail manuel. Le temps de travail est très variable suivant la taille
de la friche relevé (Flasa ~20h, Safed ~8h).

> **TODO** : Préciser le logiciel utilisé (Metashape, OpenMVG, COLMAP…) et les paramètres clés (qualité, alignement…).

---

### 3. Nettoyage PCD

**Entrée** : nuage de points brut  
**Sortie** : `PointCloud` enregistré en base, `extent_poly` calculée  
**Durée estimée** : TODO

Suppression du bruit et sous-échantillonnage au cm

> **TODO** : Décrire les filtres appliqués (suppression outliers, voxel grid…) et les seuils retenus.

---

### 4. Extraction des équirectangulaires

**Entrée** : vidéo source  
**Sortie** : images panoramiques équirectangulaires (`.jpg` — TODO : résolution cible)  
**Durée estimée** : TODO

> **TODO** : Préciser l'outil et les paramètres de projection (format de caméra, FOV…).

---

### 5. Floutage des panoramiques

**Entrée** : équirectangulaires bruts  
**Sortie** : équirectangulaires anonymisés  
**Durée estimée** : TODO

Les visages sont détectés via **Detectron2** en mode keypoint, puis floutés directement sur l'image source avant stockage, garantissant qu'aucune donnée non anonymisée n'est conservée.

---

### 6. Correction et tuilage

**Entrée** : équirectangulaires floutés  
**Sortie** : `Panoramique` enregistré en base, tuiles pour le viewer  
**Durée estimée** : TODO

Création d'un dossier par panoramique contenant les résultats :

- Correction de l'orientation des panoraniques, ajustement de l'horizon et centrage vers le nord
- Génération de la projection Cubemap
- Génération d'une image "preview"
- Tuilage de la panoramique equirectangulaire (par defaut, 2*4 tuiles)

!!! info "Arborescence de fichier"
    Pour plus d'information sur l'arborescence de fichier -> ....

---

### 7. Reprojection *(optionnel)*

**Entrée** : `Panoramique`, `PointCloud`  
**Sortie** : `Panoramique.position` et `Panoramique.orientation` renseignés (SRID 2056)  
**Durée estimée** : TODO

Géoréférence chaque panoramique dans le système de coordonnées suisse MN95
en exploitant les poses caméra issues du calcul photogrammétrique.

> **TODO** : Préciser la méthode de reprojection et les cas où elle n'est pas nécessaire.

---

### 8. Segmentation des images *(optionnel)*

**Entrée** : équirectangulaires
**Sortie** : segmentation des panoramiques  
**Durée estimée** : TODO

Utilise le modèle **Segment Anything (SAM)** pour segmenter les images panoramiques. Les

> **TODO** : Préciser le modèle SAM utilisé (ViT-H, ViT-L…), les prompts, et le workflow de validation humaine.

---

## TODO récapitulatif

- [ ] Formats vidéo acceptés en entrée
- [ ] Taux d'extraction et nommage des frames (étape 1)
- [ ] Logiciel et paramètres photogrammétriques (étape 2)
- [ ] Filtres et seuils de nettoyage PCD (étape 3)
- [ ] Outil et paramètres d'extraction équirectangulaire (étape 4)
- [ ] Modèle de floutage et seuil de confiance (étape 5)
- [ ] Format de tuilage et arborescence de sortie (étape 6)
- [ ] Méthode de reprojection et cas d'usage (étape 7)
- [ ] Modèle SAM, prompts et workflow de validation (étape 8)
- [ ] Durées de traitement estimées pour chaque étape

