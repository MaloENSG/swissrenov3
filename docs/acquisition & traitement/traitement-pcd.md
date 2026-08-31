# Traitement nuage de points

Que ce soit par lasergrammétrie ou photogrammétrie, le modèle 3D produit est un nuage de points (Pointcloud abrégé PCD). A la différence d’un modèle 3D classique sous forme de mesh (maquette BIM, modèles de jeu vidéo … etc) ou les points sont reliés pour former des faces, le PCD est un ensemble de points sans connexion. Cela signifie qu’un PCD n’est pas structuré, il n’y a aucune information sur les surfaces, la sémantique ou de notion d’intérieur/extérieur. Chaque point possède une information de position et de couleur, il est possible de lui ajouter des informations calculées mais il reste indépendant des autres points. Le PCD est une mesure brute et non structuré contrairement à un mesh qui est une donnée sujet à l’interprétation de celui qui l’a réalisé.



## Chaine de traitement automatique

Avant d’utiliser un nuage de points, celui-ci est filtré et échantillonné pour supprimer les erreurs de mesures et le rendre plus facile à exploiter.

1. **Assemblage** : Si le batiment est reconstruit en plusieurs nuages de points (plusieurs stations de laser scaner) les différents nuages de points sont assemblés et converti au format LAS.
2. **Echantillonnage** : Un sous échantillonnage par défaut de 1 cm est appliqué au PCD pour réduire le nombre de points en minimisant la perte d’information. Cette étape consiste a garder les points pour qu’il y ai un espace de minimum 1 cm entre chaque points. Un échantillonnage avec une plus grande valeur (exemple 10 cm) supprime plus de points pour ne garder que un points tous les 10 cm mais permet un traitement beaucoup plus rapide.
3. **Filtrage des points aberrants** : Il s’agit d’un filtre statistique pour supprimer les erreurs de mesures (paramètres par défaut TODO).
4. **Alignement** : Un algorithme permet de calculer l’angle de rotation du nuage de points (donc du batiment) par rapport au nord. Il est ensuite possible d’aligner le batiment avec le nord dans un référentiel local. Ce paramètre est sauvegardé dans les Info du PCD (voir structure PCD).
5. **Classification** : Un classifieur RANDLANet(hyperlien) permet de classifier automatiquement chaque point du nuage pour enrichir la donnée (voir section Classif auto).

## Classification automatique (optionnel)

Afin d’enrichir les informations du nuage de points, il est possible de classifier chaque point individuellement pour lui attribuer une classe. Par exemple, pour un PCD modélisant une pièce, certains points sont classés comme étant « sol », d’autres seront « fenêtre » … etc. Cet enrichissement constitue une première étape pour amener le nuage de points brut vers une donnée interprété et ainsi se rapprocher du mesh. La classification automatique, bien qu’imparfaite, permet de classifier plusieurs dizaines de millions de points (soit plusieurs centaines de m2) en quelques minutes sans intervention humaine. Cela permet par la suite de sélectionner des morceaux du PCD selon la nature des points et d’appliquer des traitements plus complexes. 

Par exemple, un architecte souhaite modéliser les murs d’un bâtiment pour sa maquette BIM. Le nuage d points qui l’intéresse représente 100 millions de points (plusieurs Go) et les murs ne sont pas toujours faciles à voir à cause des portes, des encombrants et des structures au plafonds. Le PCD est classifié donc il suffit à l’architecte de sélectionner les points « mur » pour ne garder que ce qui l’intéresse. Il obtient un PCD plus léger et plus lisible pour travailler.
Les classes retenues dans le cadre du projet Swissrenov sont les suivantes : Plafond - Sol - Mur - Poutre - Colonne - Fenêtre - Porte - Encombrant - Escalier.

Une classification donne une information sur chaque point mais ne permet pas de répondre à certaines questions comme « Combien y a-t-il de fenêtres dans ce bâtiment ? » ou « Quelle est le volume d’encombrant dans cette pièce ? ». Mais une classification de bonne qualité permet d’utiliser des algorithmes (Voir Clusturing, vectorisation TODO) et de répondre à ces questions avec une suffisamment bonne exactitude.

## Logiciels et formats de fichiers

Les outils pour manipuler un nuage de points :

-	**Cloudcompare** : gratuit et open source. C’est la référence de base pour la manipulation et le traitement de PCD. Utilisable en ligne de commande pour de l’automatisation.
-	**Meshlab** : Logiciel gratuit et open source. Très utile pour des traitements spécifiques et du maillage.
-	**Laspy** et pyE57: Librairies python libres. Très pratique pour ouvrir des fichiers LAS et E57.
-	**Open3D** : Librairie python open source. La référence en développement python pour le traitement de nuage de points. Souvent utilisé avec Laspy et pyE57.
-	Demander a marlon outil archi
-	**Cyclone 3DR** : Logiciel propriétaire Leica. Spécialisé pour le traitement topographique.

Les formats des fichiers de nuage de points :

- **LAS/LAZ** : standard LiDAR (ASPRS). LAS non compressé, LAZ compressé (~10:1). Attributs : XYZ, intensité, classification, RGB. Référence en topographie et scan aérien.
- **E57** : format ouvert ISO, conçu pour l'interopérabilité entre scanners terrestres. Supporte plusieurs scans et images panoramiques dans un seul fichier.
- **PLY** : format simple et flexible (Stanford). Existe en ASCII et binaire. Très utilisé en recherche et vision 3D, sans notion de classification LiDAR.


## Structure d’un PCD en sortie 

Après traitement, un PCD possèdes 5 attributs structuré de manière à facilité des export, calculs ou traitements supplémentaire.

-	**Géométrie** : Un points 3D possède des coordonnées (X, Y et Z) dans le système Suisse (EPSG : 2056) ou dans un système local associer au système Suisse par des paramètres stocké dans les Informations.
-	**Couleur** : Elle est décrite par 3 composantes (Rouge, Vert et Bleu) et permet d’associer une couleur à un point.
-	**Classification automatique** : 
-	**Indexation** : Chaque point est « numéroté » pour pouvoir le retrouver efficacement dans la reprojection 3D.
-	**Informations** : Il ne s’agit pas d’une information par point mais plutôt de métadonnées associées au PCD. Celui-ci peut subir des transformations (changement de référentiel) ou filtrage (échantillonnage, découpe … etc) qu’il faut conserver pour assurer la traçabilité de la donnée.  
