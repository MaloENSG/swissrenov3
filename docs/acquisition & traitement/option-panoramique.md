# Images panoramiques enrichies

Une fois les panoramiques construites, plusieurs étapes optionnelles sont possibles pour enrichir la donnée image. Ces étapes sont complètement automatiques mais représentent un temps de calcul important.

## Traitements optionnels

Ces étapes optionnels sont indépendantes les une des autres et sont à adapter suivant le résultat souhaiter. Elles peuvent être réalisées plusieur fois avec un paramétrage différent.

### Reprojection 2D/3D

La reprojection ou connexion 2D/3D consiste a lier une image dont on connait la position dans un référentiel avec un élément 3D dans le même référentiel. Chaque point du nuage de points 3D est projeté sur une image pour connaitre sa position dans celle ci. La reprojection 3D est l'étape la plus longue et nécéssite beaucoup de calculs. Des optimisations sont possibles mais ce processus ne peut être réalisé en temps réel.

Les différents produits de cette reprojection sont :

- **La carte d'indexation `mapping.tif`** : Résultat le plus important de ce traitement, la carte d'indexation est une "image" particulière qui permet de sauvegarder les points visibles dans l'image panoramique et de conserver leur identité. Ce résultat permet de réaliser une reprojection de manière optimisé et en temps réel sans reproduire les calculs lourds.
- **La carte de position `position.tif`** : La reprojection permet d'associer un pixel de l'image à un point 3D. Celui ci possède une géométrie X, Y et Z. La carte de position enregistre la géométrie du point dans le pixel correspondant. Au lieu de posséder 3 attributs RVB, ce pixel possèdes 3 attributs XYZ. Cela permet d'avoir une géométrie dans l'image sans repasser par le nuage de points. ⚠️ Attention, ce résultat est très volumineux (~5 Mo par panoramique) et n'est pas toujours pertinent.
- **La carte de profondeur `depthmap.tif`** : Il s'agit d'un résultat intermédiaire pour calculer l'occlusion du nuage de points 3D, c'est à dire qu'est ce qui est visible et qu'est ce qui est caché par un élément au premier plan. Un point 3D qui se trouve dans une autre pièce n'est normalement pas visible car caché par un mur, cette carte de profondeur permet de filtrer les points derrière le mur en conservant les points du mur qui sont effectivment visibles.
- **Les métadonnées `metadata.json`** : Ce fichier conserve des informations essentiels pour garantir la traçabilité du traitement. 

*vitesse de traitement : 100 img/h avec 100 Millions de points*

!!! note "Détail technique"


### Segmentation des images

Les images sont automatiquement segmentées, c'est à dire découpées de manière cohérentes par objet. Par exemple, pour une image prise dans une pièce avec des fenêtres et des caisses entreposées, la segmentation permet de découper chaque caisse et chaque fenêtre dans l'image. Cette pré-analyse de l'image ne permet pas de qualifier la nature de l'objet, on ne pourra pas dire automatiquement ceci est une fenêtre. Mais cela permet d'intéragir plus facilement et rapidement avec l'image dans la visite virtuelle.

*vitesse de traitement : 200 img/h*

### Structure et sauvegarde



## Ajouter un enrichissement

La structure et l'arborescence d'une `panoramique` permet d'ajouter autant de traitement sur l'image que l'on souhaite. Il est possible par exemple d'ajouter une détection/segmentation d'objet spécifique en utilisant un modèle d'IA "classique" comme YOLO ou SegFormer. Ces modèles sont dit "classique" car ils sont concu pour traiter des images normals, mais la projection Standard Cubemap (SC) permet de s'affranchir en parti des problèmes (de déformation) liée aux images panoramiques pour exploiter ces modèles déjà entrainé, répendus et très performants.

D'autres possibilités pourrait être de projeter une idée d'aménagement sur l'image ...