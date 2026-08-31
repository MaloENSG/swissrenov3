# 2.Photogrammetrie - Les concepts

---

## Extraction des fisheyes

### Projection brut et fisheyes

Une caméra panoramique utilise une projection pour enregistrer les photos et vidéos des deux objectifs. Chaque objectif produit un flux vidéo distinct. Certaines caméras utilisent la projection Standard Cubemap tandis que d'autres utilisent une projection Equirectangulaire ou directement des fisheyes. Dans le cadre de ce projet, la caméra est une Gopro Max 360 qui utilise la projection Equi-Angular Cubemap (EAC). La plupart du temps, il faut donc transformer la projection des images en fisheyes pour les rendres exploitables par un logiciel de photogrammétrie. Il existe aussi plusieurs types de projection fisheyes (stéréographique, équidistante, orthographique). Il faut vérifier quelles sont les projections fisheyes géré par le logiciel de photogrammétrie utilisé.

| Projection | Aperçu | Distortion | Résolution | Usage |
|---|---|---|---|---|
| Équirectangulaire (EQ) | ![logo](images/logo.png) | Forte aux pôles | Moyenne | Standard VR, très interopérable |
| Équi-angulaire Cubemap (EAC) | ![logo](images/logo.png) | Faible et uniforme | Excellente | Répendu, Youtube 360 |
| Cubemap standard (SC) | ![logo](images/logo.png) | Faible (bords visibles) | Bonne | Répendu, jeu vidéo |

!!! warning "Important" 
    Il est impératif de vérifier ce que la caméra utilise comme projection et ce que le logiciel de photogrammétrie à besoin pour adapter cette étape au matériel utilisé.

### Calibration & modèle de caméra

Les fisheyes sont des images avec une distortion importante. C'est à dire que l'image est déformé par rapport à la réalité. De plus les images fonctionnent par paire avec un bras de levier (géométrie constante entre les deux objectifs). Le processus de photogrammétrie néccésite un modèle de caméra pour produire une reconstruction efficace et précise. Ce modèle de caméra est un ensemble de paramètres qui permet de modéliser le comportement de la caméra. 

Lorsque l'on souhaite utiliser une nouvelle caméra panoramique pour réaliser un relevé, il faut au préalable déterminer cet ensemble de paramètres avec rigueur, c'est ce que l'on appele une calibration. Chaque caméra est unique et possède son propre modèle.

### Découpage vidéo

!!! info "Paramètres utilisés"
    - **Caméra** : Gopro Max 360
    - **Résolution** : 5.6K
    - **Frames per second (FPS)** : 30 FPS
    - **Fréquence** : 2 frames par second (1 frame sur 15)

---

## Principe de la photogrammétrie

### Points remarquables & correspondance

La première étape consiste à trouver des points remarquables dans les images. Chaque image est analysé pour trouver des points suffisement distinctifs 

Ensuite, ces points remarquables sont comparé avec les points dans les autres images pour trouver des correspondances. Pour cela, deux points sont comparé en tenant compte des pixels dans leur voisisnage. Si la similitude est suffisemment importante. alors les deux points sont mis en correspondances. Après la détermination des correspondances entre points remarquables, chaque image est comparé avec toutes les autres images. Si deux images ont de nombreux points homolgues, et que la position relative de ces points est cohérente, alors les images sont misent en correspondance l'une avec l'autre.

### Recouvrement entre images

Dans un problème de photogrammétrie classique, avec des images "normales", les images ont un champs de vision limité. Si l'on souhaite avoir deux images qui se correspondent, il faut avoir suffisemmment de points homologues pour appareiller ces deux images donc avoir suffisemment d'éléments présent à la fois sur l'une et l'autre image. On appelle cette contrainte le recouvrement. 
Dans le cas ou un élément n'est visible que sur une seule image parce que le cadrage des autres images ne permet pas de voir cet élément, il ne sera pas reconstruit. On réalise assez facilement les contraintes qu'impose le recouvrement pour avoir une reconstruction exhaustive d'une pièce. Il faut prendre énormément de photos et faire attention à bien couvrir toutes les surfaces. Avec un appareil photo classique, cela devient un travaille long et fastidieux.

Avec une caméra panoramique, le champs de vision n'est pas limité. Donc plusieurs images prisent dans une même pièce trouverons nécessairement des points homologues (des éléments similaires pour se rattacher). Il n'y a plus besoin de se préoccuper du cadrage des photos mais seulement de là ou elles sont prisent. De plus, la caméra en mode vidéo enregistre en continu donc les images se suivent, il n'y a pas d'interruption entre 2 images qui serait prisent à deux endroits complètement différent. Une image prise après une autre est suffisement proche pour trouver des éléments similaire mais ne se trouve pas exactement au même endroit ce qui est idéale pour une reconstruction par photogrammétrie.

### Positionnement relatif/absolu



### Résultats

La reconstruction par photogrammétrie produit plusieurs résultats très utiles pour analyser un batiment :

- **Nuage de points dense** : 
- **Position et orientation caméras** :

Plusieurs produits dérivés de ces résultats sont aussi très utiles :

- **Rasterisation** : Le nuage de points 3D neccésite des outils particuliers et une bonne puissance de calcul pour pouvoir être exploité. La rasterisation consiste à produire une images plan à partir de ces points pour former une vue en coupe ou une vue en élévation.
- **Visite virtuelle** : A partir des positions et orientations des caméras, il est possible de réaliser tout un tas de mesures. Dans notre cas, elles servent à construire une visite virtuelle du site et à connecter les images 2D avec un nuage depoints 3D.

!!! info "A développer"
    - **Orthomosaic** : C'est exactement le même principe que la rasterisation. Mais au lieu d'utiliser le nuage de points ce qui fait perdre en résolution, on utilise une transformation des images pour obtenir une image plan avec une exellente résolution. Ce traitement nécessite des conditions particulière pour fonctionner.




