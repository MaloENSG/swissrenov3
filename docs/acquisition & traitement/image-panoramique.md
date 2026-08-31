# Images panoramiques

Pour exploiter les images panoramiques, il faut utiliser une projection, c'est à dire une déformation pour transformer une images sphérique à 360 degrés sur une surface 2D (Comme pour un globe terrestre en cartographie que l'on transforme en carte monde). 



---

## Grandes étapes

Plusieurs étapes sont nécessaires pour passer d'une vidéo à une visite virtuelle finie. Toutes ces étapes sont entièrement automatisées et ne nécessites aucunes interventions.

### Extraction des equirectangles

Comme pour la reconstruction 3D par photogrammétrie, une vidéo (séquence) est découpée en suite d'images avec la même fréquence. Cette fois si, les images des deux objectifs de la caméra panoramique sont assemblées pour former une image equirectangle (EQ) pratiquement prète à être exploité. La série d'images EQ est stocké dans un dossier séquence.

*vitesse de traitement : 1400 img/h*

### Floutage des visages

Pour garantir l'anonymat de l'opérateur et des personnes présentes sur le site pendant l'acquisition, un floutage automatique des visages est appliqué sur les images EQ qui sont écrasée. Les images avant floutage ne sont donc pas conservées. Le floutage automatique est réalisé avec un Detectron2 en mode keypoint.

*vitesse de traitement : 3200 img/h*

!!! info 
    Il est possible d'ajouter d'autres algorithmes pour flouter les plaques d'immatriculations ou les éléments sensibles lors de cette étape.

### Correction et tuilage

Les images sont prises dans une orientation arbitraire, donc l'horizon n'est pas parfait (la caméra n'est jamais complètement à l'horizontal pendant l'acquisition). Il faut appliquer une correction de l'orientation pour corriger l'image. L'orientation et la position réelle de chaque image est calculé lors de la première étape de photogrammétrie. Cette information d'orientation calculé permet ensuite d'appliquer la correction sur l'image. Par la même occasion, le centre de l'image est orienté vers le nord pour facilité l'exploitation de l'image et la connexion entre l'image et la 3D.  

Une fois corrigé, chaque pixel de l'image correspond à une orientation dans le monde réelle. La ligne centrale est l'horizon, le centre correspond au nord et les deux coté de l'images (bord droit et bord gauche) se rejoignent au Sud. Les bords supérieur et inférieur correspondent respectivement au dessus et dessous. Ces bords ne sont pas visibles dans un visualisateur d'image panoramique.

Une image panoramique EQ est assez volumineuse (5 à 10 Mo pour une bonne résolution) ce qui peut ralentir une page web pour l'affichage. L'échange de données est fluidifié grace au tuilage de l'image. Celle ci est découpée en tuile pour permettre à une page web de charger progressivement l'image. Cette notion de tuile est très souvent visible sur les applications de cartographie (tester géoadmin.ch avec une connexion assez lente). Ces tuiles sont aussi compressées au format WEBP ce qui améliore encore davantage la rapidité du visualisateur web. 

!!! warning "Attention"
    Toute modification d'image doit être réalisé avant cette étape de tuilage qui fragmente l'image ce qui complique énormément la mise à jour

### Structure et sauvegarde



---

## Combinaison de 2 projections pour 2 usages

La projection Equirectangle (EQ) est la plus répandu pour l'échange et la visulalisation d'images panoramiques. C'est aussi un format facile à exploiter pour des traitements avec de la 3D
car très simple dans sa définition mathématique (TODO source wikipédia). 
Cette projection présente cependant certains défauts comme une résolution très hétérogène entre le centre et les bords inférieurs et supérieurs de l'image ainsi que des déformations très importantes sur celles ci.

Certains traitements particuliers comme l'analyse par outil IA peuvent être limité par la projection EQ. C'est pourquoi elle est combinés avec une autre projection, la Standard Cubemap (SC).

### Optimisation de la visualisation



### Projection pour l'analyse IA

La projection Standard Cubemap (SC) consiste a utiliser un cube pour construire l'image panoramique. Un cube est composé de 6 faces donc concrètement, l'image panoramique est dans ce cas un ensemble de 6 images différentes. Ces images comportent peu de déformations et sont très similaires à des images perspectives (images classiques), elle n'ont visuellement pas vraiement de différence. 
Cette absence de déformation permet à des modèles d'IA d'analyser beaucoup plus facilement les images pour en extraire automatiquement diverses informations. Les images plus petites du SC permettent de saisir beaucoup plus de détailles que pour une grande image EQ.

L'intéret de la projection SC par rapport à un ensemble d'images perspectives classiques est de pouvoir repasser très facilement vers une projection EQ, et donc de réutiliser les outils de visualisation et de connexion avec la 3D. 
