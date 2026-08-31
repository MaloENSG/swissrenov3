# 1.Acquisition

## Méthodes et matériels

Pour réaliser la numérisation d'un terrain, bâtiment ou autre site, il existe principalement 2 méthodes :

- **La reconstruction par photogrammétrie** nécessite des photos ou vidéos. Il est possible d'utiliser cette technique avec n'importe quel appareil photo ou caméra. Mais pour simplifier l'acquisition et gagner en efficacité, nous utilisons une caméra panoramique avec un champ de vision à 360 degrés. Cette caméra prend des photos/vidéos avec un champ de vision qui n'est pas limité comme une caméra classique, ce qui permet de ne pas perdre de temps avec les problèmes de recouvrement habituels en [photogrammétrie](photogrammetrie.md). Le relevé par caméra panoramique consiste à prendre une vidéo en se déplaçant. Cette vidéo est ensuite découpée en images à intervalle de temps régulier pour produire une séquence d'images.

- **La reconstruction par lasergrammétrie** nécessite un laser scanner. Il s'agit d'un appareil de mesure spécialisé pour la réalisation de relevés 3D. Le relevé par lasergrammétrie consiste à installer l'appareil à un endroit pour réaliser une acquisistion puis à déplacer l'appareil à un autre endroit. Cette méthode est intégralement réalisé par un bureau expert dans le domaine.

---

## Acquisitions caméra panoramique

La caméra est idéalement fixée sur un casque sur la tête de l'opérateur ou fixée sur une perche à selfie. En raison de son champ de vision dans toutes les directions (360 degrés), la caméra filme nécessairement l'opérateur pendant l'acquisition. L'opérateur devient donc un masque d'obstruction sur une partie du champ de vision de la caméra.

- **Fixation sur casque** : la partie obstruée correspond au sol, il n'y aura donc pas d'obstruction sur les éléments autour, mais la reconstruction et la visualisation du sol seront de moins bonne qualité. Cette solution peut poser problème lorsque la hauteur sous plafond est basse, comme dans des combles avec des poutres de charpente ou lors du passage d'une porte.

- **Fixation sur perche à selfie** : la fixation et la manipulation de la caméra sont plus simples. L'opérateur tient la perche à bout de bras de manière à se tenir le plus loin possible de la caméra. En s'éloignant, l'opérateur occupe une surface plus faible du champ de vision et représente une obstruction plus faible. Cette solution est intéressante dans les passages avec peu de hauteur ou pour passer au-dessus des meubles et machines.

### Principe de cheminement

Une acquisition avec une caméra fonctionne de « proche en proche » : l'image acquise à l'instant T est rattachée à l'image acquise à l'instant T-1 de manière successive lors de la reconstruction 3D. Ce concept est appelé un **cheminement**.

Une erreur négligeable peut exister entre 2 étapes successives sans être observable localement. Mais cette erreur s'accumule au fur et à mesure et devient problématique au-delà de 20-30 mètres (erreur d'environ 10 cm à partir de 20 mètres).

Pour limiter cette accumulation d'erreurs, il faut parcourir la zone en réalisant des **boucles** : on termine la vidéo là où on l'a commencée, ce qui permet aux étapes de fin de se raccrocher aux étapes du début. On appelle cela un **cheminement fermé**. Cette méthode ne fait pas disparaître l'erreur mais la minimise et facilite grandement la reconstruction 3D.

### Cheminements primaires et secondaires

Concrètement, il n'est pas possible de faire des boucles partout. L'opérateur doit reconnaître le site avant l'acquisition pour identifier les boucles de cheminement. Ces boucles constituent le squelette de la reconstruction 3D : elles doivent passer par les couloirs et les axes principaux du site sans considérer les petites pièces autour — comme des autoroutes qui ne s'arrêtent pas à chaque ville mais constituent des axes auxquels se rattachent des routes plus petites.

Les pièces et zones autour d'une boucle font l'objet d'un **cheminement secondaire**, qui se rattache aux boucles de **cheminement primaire**.

Lors d'un cheminement secondaire, l'opérateur parcourt la pièce en fonction de sa forme, des encombrements et des ouvertures :

- **Pièce allongée sans ouverture et de moins de 20 m** : un aller-retour est idéal, un aller simple convient aussi.
- **Pièce avec plusieurs portes** : une boucle est toujours idéale. Sinon, il est possible de rattacher le début et la fin du cheminement à des cheminements primaires déjà réalisés.

---

### Cas pratiques

#### Cas pratique 1 : impasse

Une impasse est une partie d'un site qui ne possède qu'une seule entrée/sortie et qui nécessite un cheminement de plus de 20-30 mètres pour être couverte — autrement dit, une partie suffisamment importante pour un cheminement primaire mais qui ne permet pas de faire une boucle.

- **Impasse ouverte** : il est possible de voir, depuis un point de l'impasse, une zone à laquelle se rattacher (fenêtre ou autre ouverture, point de coordonnées connu). Il n'est pas possible de circuler mais le rattachement au reste du bâtiment est facile.

- **Impasse fermée** : il n'est pas possible de voir une zone connue à laquelle se rattacher. Cela peut arriver dans un sous-sol ou des combles. Il faut alors accepter de possibles erreurs sur cette partie de l'acquisition (un long couloir rectiligne peut finir courbé dans la reconstruction). Autrement, il faut mesurer des points de référence (avec un théodolite ou un GNSS si en extérieur) pour s'y rattacher.

#### Cas pratique 2 : passage étroit

Un passage étroit est un passage qui restreint la visibilité : ouverture (porte), escalier, couloir étroit, ou pièce avec beaucoup d'encombrements (étagères hautes et faiblement espacées).

Il est préférable de **ralentir le cheminement** dans ces espaces pour augmenter le nombre de photos et faciliter la reconstruction 3D. Faire particulièrement attention aux virages serrés (escalier, angle droit dans un couloir étroit) en marchant très doucement, car la visibilité très réduite pose souvent problème.

#### Cas pratique 3 : changement d'étage

Un changement d'étage est délicat car une cage d'escalier est souvent un passage étroit (voir cas pratique 3) et constitue le seul point de passage entre étages. Si le bâtiment possède plusieurs escaliers, il faut réaliser une boucle qui passe par ces escaliers.

En général, il n'y a qu'un seul passage : un étage peut alors être considéré comme une impasse (voir cas pratique 2). La solution consiste à faire un relevé proche des fenêtres (voire à passer la perche à selfie à l'extérieur) pour rattacher le cheminement de l'étage aux cheminements extérieurs.

---

### Matériaux et luminosité

La reconstruction 3D par photogrammétrie repose sur des photos. Si la luminosité ambiante est trop faible lors de l'acquisition, il n'est plus possible de distinguer correctement les détails et la reconstruction 3D est dégradée, voire impossible.

Normalement, un éclairage artificiel est suffisant — il faut veiller à allumer les lumières avant l'acquisition (attention aux éclairages automatiques).

Les combles sous certaines toitures ainsi que certains sous-sols peuvent ne pas avoir d'éclairage. Dans ce cas, **la méthode par caméra n'est pas possible** : il faut utiliser un laser scanner, qui n'est pas impacté par la luminosité ambiante.

---

## Acquisition laser scanner

!!! warning "A Compléter"
