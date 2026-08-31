# Classification automatique

## Pourquoi classifier la donnée ?

Le nuage de point est une donnée brute, c’est-à-dire qu’il modélise une mesure physique et ne possède pas de structure ou de sémantique. On ne peut à priori pas dire a quoi correspond un point seul, c’est le contexte des points autour qui nous permettent de déterminer ce à quoi il correspond. La classification automatique permet de donner du sens a chaque points et nous aide a filtrer, analyser et extraire d’avantage d’informations.

## Principes de bases & nomenclature

Une classification automatique de points 3D consiste à faire passer le nuage de points dans un modèle d’IA pour associer une classe à chaque point. Les classes possibles sont limitées (généralement une dizaine de classes) et à définir lors de la création du modèle d’IA. Une fois le modèle entrainé, il attribuera une classe parmis celles qu’il a apris lorsde son entrainement aux points du nuage et ne pourra pas en trouver de nouvelles.
Un modèle d’IA n’est donc utile que s’il est capable de classifier des éléments pertinents. C’est pour cela qu’avant d’entrainer un modèle, il est important de définir les classes dont nous avons besoins et des problèmes que cela pourra résoudre. 
Définir les classes est la première étape majeure dans ce processus. C’est ce qui va guider la suite du travail. Cette définition s’appelle la nomenclature et doit permettre de répertorier les classes les plus utiles à la résolution de notre problème tout en restant cohérent avec les contraintes liées à la donnée. Par exemple, il parait évident qu’une classe « canapé » n’est pas du tout intéressante pour la rénovation d’un batiment. A l’inverse, des classes trop spécifiques ne pourrons pas être apprises correctement par le modèle d’IA. Faire la différence entre un mur porteur et une cloison n’est pas possible visuellement, donc un nuage de points basé sur la géométrie et la couleur ne permettra jamais de distinguer les deux.

## Constitution d’un jeu de données

Dans le cadre du projet Swissrenov, le jeu de données FRICHE-3D (Fragment Recognition In CHaotic Environments) à été créer pour entrainer un modèle spécialisé dans les éléments de constructions. En raison d’un manque initial de données pour un entrainement complet, FRICHE-3D repose sur S3DIS pour la classification d’intérieur. Cela permet d’exploiter des modèles déjà pré-entrainé pour améliorer et accélérer considérablement les résultats de notre modèle. L’intérêt de ce nouveau jeu de données est de pouvoir entrainer le modèle avec des données qui comportent des artéfactes et des configurations typiques d’une friche indistrielle tandis que S3DIS est un jeu de données qui modélise des bureaux de manière exhaustif.

> Nomenclature des classes du jeu de données **FRICHE-3D**.

| ID | Nom | Description |
|----|-----|-------------|
| `0` | Plafond* | Plafond plat, en pente, sous les toits |
| `1` | Sol* | Sol, planchers, rampe d'accès peu raide |
| `2` | Mur* | Mur, cloison |
| `3` | Poutre* | Élément horizontal, charpente de toiture |
| `4` | Colonne* | Élément vertical |
| `5` | Fenêtre* | Ouverture sans passage |
| `6` | Porte* | Ouverture avec passage |
| `7` | Encombrant** | Meubles, équipements, objets ne constituant pas le bâtiment |
| `8` | Escalier | Escalier, rampe raide |
| `99` | No label | Éléments non labélisés |


**Classes identiques à S3DIS*

***Encombrant réunit toutes les autres classes de S3DIS*

---

### Labélisation manuel

La labéilsation des données est réalisé avec Cloudcompare et se décompose en 3 étapes. D’autres méthodes sont possibles du moment que le résultat est un nuage de points avec un attribut de labélisation qui respect les identifiants indiqués dans la nomenclature.
-	Découpage par niveau : Dans le cas ou le batiment comporte plusieurs niveaux, celui-ci est découpé par niveau.. Lors de cette étape, le batiment est aussi découpé par ensemble homogène (hangar séparé d’une zone bureau par exemple). Un nuage de points modélisant un batiment (environ 50-100 millions de points) est séparé en plusieurs sous-nuages (environ 5-10 millions de points). Chaque sous-nuage constitue un nuage de points à labéliser séparemment.
-	Labélisation des classes : Les sous-nuage sont découpés par classe et suvegardés. Chaque sous-nuage est donc framenté en 10 fichiers maximum (car il y a 10 classes). Les éléments les plus volumineux sont découpés en premiers : sol – mur + fenêtre – plafond. Ensuite les plus petits éléments sont découpés : fenêtre et mur – poutre – porte … etc. 
-	Assemblage : un script python s’occupe pour chaque sous-nuage de prendre les différents fichiers LAS pour les assembler en ajoutant l’attribut de labélisation `label`.

### Format du jeu de données 

Le jeu de données FRICHE-3D est structuré en plusieurs fichiers LAS avec l’arborescence suivante :

```
FRICHE-3D/
└── site1/
    ├── pcd-name/
    │   ├── pcd-name_01.las
    │   ├── pcd-name_02.las
    │   └── ...
    └── ...
```

Pour l’utiliser en entrainement ou en inférence avec un modèle d’IA, il faut le pré-traiter pour le mettre sous une forme exploitable par ce modèle.

!!! warning "a finaliser"
    FRICHE-3D est convertible au format S3DIS via le script `convert_S3DIS.py`.

## Entrainement & inférence

Entrainer un modèle d’IA demande beaucoup de puissance et de temps de calcul. Toutes les entreprises, surtout de petite taille, ne peuvent pas toujours s’équiper en GPU et unité de calcul onéreux. Il y a deux architectures qui semblent particulièrement interssantes car elles permettent d’avoir de très bon résultats (parmis les meilleurs dans le domaine) tout en nécessitant très peu de puissance de calcul (comparé aux autres).
Tableau synthétique RandLANet – SPT

### Entrainement RandLANet

Un premier entrainement est réalisé avec l’architecture RandLANet car plus simple à prendre en mains. De plus, il n’y a pour le moment que très peu de données labéliser ce qui pourrait ne pas suffire pour une architecture plus complexe.

### Paramètres d’entrainement



### Inférence et post-traitement

Comme pour l’entrainement, l’inférence a besoin de données prétraitées pour fonctionner correctement. Mais après une inférence, on veut retrouver un nuage de points cohérent et géoréférencé. Lors de la conversion des données au format S3DIS, les paramètres de transformation sont enregistrés pour chaque sous-nuage. Il faut réappliquer ces transformations sur le résultat de l’inférence pour retrouver des sous-nuages géoréférencés (et classifiés). Il ne reste plus qu’a les assembler de nouveau pour obtenir un nuage complet prêt à être intégré dans un viewer.
