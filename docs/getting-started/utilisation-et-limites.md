# Utilisation et limites

## Nuage de points

### Généralités

Un nuage de points est un ensemble de points 3D auxquels il est possible de rattacher des informations supplémentaires comme la couleur ou d'ajouter une estimation automatique de nouvelles informations comme la sémantique. Il s'agit d'une donnée non structurée donc chaque point est indépendant des autres et n'a à priori aucun lien avec les autres. C'est à l'humain d'interpréter cet ensemble de points pour reconnaître des géométries et des informations. La couleur et la sémantique sont des informations supplémentaires pour l'aider dans son interprétation.

### Utilisation en pratique

Le nuage de points est très pratique pour faire une analyse géométrique. Il est possible de mesurer une coordonnée ou une distance entre 2 points. Il est aussi possible de mesurer des surfaces et des volumes avec une très bonne précision. Toutes ces mesures géométriques sont difficiles à automatiser, car il faut d'abord comprendre ce qu'on mesure, où et comment. Donc malgré une précision des points 3D au mm, cette tâche repose en premier lieu sur une interprétation humaine.

Assembler plusieurs nuages de points modélisant plusieurs pièces différentes permet de mettre en évidence les épaisseurs. On peut donc appréhender l'épaisseur entre les deux surfaces d'un même mur ainsi que son volume, mais il ne sera pas possible d'en connaître la composition !

Il est aussi possible d'exporter un nuage de points directement dans un logiciel de modélisation (Revit, AutoCAD) pour redessiner par-dessus.

### Produit dérivé

La donnée 3D est souvent difficile à exploiter car elle demande des ressources informatiques importantes et des outils particuliers. Pour faciliter son utilisation, il est possible de faire une conversion en image (coupe raster en vertical ou horizontal). Les points 3D sont mis dans une image de manière à obtenir un plan de la scène. Cette opération dégrade légèrement la qualité initiale des points 3D (précision au cm) mais les mesures restent bonnes et sont en pratique plus faciles à réaliser sur l'image que sur les points 3D. Les épaisseurs et certaines surfaces sont plus évidentes.

Les points 3D peuvent aussi être enrichis par des informations comme la sémantique. Pour plus d'informations, consulter la section TODO.

### Limites

Suivant la méthode d'acquisition utilisée (photogrammétrie ou lasergrammétrie), le résultat est plus ou moins exhaustif. Les absences de données par endroit peuvent complexifier l'interprétation et compromettre les mesures géométriques. De plus, la précision des points 3D dépend aussi du matériau qui est modélisé, les surfaces vitrées, brillantes et réfléchissantes ont toujours une précision très faible et ne sont pas facilement mesurables.

Des erreurs de mesures sont toujours présentes dans un nuage de points 3D, ce qui peut rendre les mesures géométriques compliquées dans une interface 3D. Les mesures sur des objets (meubles, petits éléments, etc.) ne sont pas très précises. Il vaut mieux travailler à plus grande échelle comme une pièce ou un étage.

Enfin, la mesure dépend aussi de la densité du nuage de points (1 point tous les cm ou tous les 10 cm ?). Une faible densité ne permet pas de faire une mesure précise si l'objet mesuré est petit. Par exemple un point tous les 10 cm suffit à obtenir les dimensions d'une pièce mais pas celles d'un meuble.

### Exemple

L'objectif est de mesurer les surfaces d'une façade pour chiffrer les quantités. Dans le nuage de points 3D, il suffit de cliquer deux points aux extrémités de la façade pour relever précisément la longueur totale. Toutes les parties de la façade sont faciles à mesurer avec une bonne précision. Une image panoramique n'aurait pas été adapté car les effets de perspectives rendent la mesure diffile et très imprécise.

### Pour récapituler

- La géométrie est très précise et permet de réaliser des mesures au cm sans problème voire au mm dans certains cas.
- Les points 3D peuvent être convertis en différents produits comme des plans images (raster) pour faciliter leur utilisation.
- C'est une donnée particulièrement adaptée pour travailler à l'échelle d'une pièce, d'un étage ou d'un bâtiment.
- La méthode d'acquisition d'un nuage de points ne permet jamais de « voir » un élément caché ou obstrué.
- Les mesures semi-automatiques sur des objets sont davantage des estimations et non des vérités absolues.

## Images panoramiques

### Généralité

Une image panoramique est une image qui « regarde » dans toutes les directions. Elle n'a donc pas de problème de cadrage. Elle permet d'explorer un environnement avec un regard d'humain qui tourne autour de lui pour avoir une vision exhaustive (à la différence d'un plan ou d'une carte). Il n'y a à priori aucune géométrie mais l'interprétation de l'image est évidente.

### Utilisation en pratique

Une panoramique peut être combinée avec la 3D pour permettre d'ajouter la géométrie. Cette combinaison est très intéressante pour de nombreuses applications mais la qualité des mesures géométriques reste bien moins bonne qu'avec un nuage de points 3D. Les mesures géométriques dans la visite virtuelle permettent d'avoir une idée des dimensions des éléments que l'on visualise. Ces images sont surtout idéales pour visiter et reconnaître des petits détails (interrupteurs, grosses fissures, matériaux, etc.).

L'image s'avère idéale pour identifier et dénombrer les petits éléments de la friche (éléments de construction, meubles, tuyauterie) et les localiser facilement à l'aide de la 3D combinée.

### Produit dérivé

Les images panoramiques brutes sont exploitées dans une visite virtuelle ou dans des logiciels spécialisés. Cependant les informations issues de cette donnée sont faciles à convertir.

Le premier produit dérivé est tout simplement une capture d'écran d'un point de vue particulier. Cela permet de produire une illustration d'une pièce, d'un espace, d'une façade ou d'un élément quelconque en toute simplicité.

Ensuite, la visite virtuelle permet de saisir des informations sur les images. Ces informations peuvent être exportées pour réaliser des fiches signalétiques, des cartes de localisation des objets ou un fichier de tableur (type Excel).

### Limites

Comme pour l'acquisition de points 3D, ce qui n'est pas directement visible n'apparaît pas dans les données. Un élément caché ne sera donc pas mesurable ou identifiable. De plus, si la luminosité est faible, il peut y avoir du flou dans les images.

Plus le sujet est loin, plus sa précision dans l'image est réduite. S'il n'y a pas d'image plus proche, il n'est parfois pas possible d'analyser certains éléments car la précision ne le permettra pas.

### Exemples

L'objectif est de mettre à jour les plans d'un site avec de nombreux escaliers. Grâce aux images panoramiques, il est possible de faire une photo d'illustration pour chaque escalier et de déterminer le type. Escalier en colimaçon, à limon droit ou en courbe, avec ou sans contremarches : toutes ces informations visuelles sont accessibles sur les images. Il est aussi possible de compter le nombre de marches et de voir les matériaux. Un formulaire adapté permet de saisir facilement et rapidement toutes ces informations de manière structurée.

Pour ce qui concerne la géométrie de l'escalier, il faut utiliser les nuages de points 3D pour extraire précisément les mesures.

### Pour récapituler

- Les images panoramiques offrent une vision très concrète du site, avec une interprétation immédiate et intuitive.
- Les petits éléments (< 1 cm) sont facilement visibles.
- C'est idéal pour repérer, dénombrer, annoter des éléments de petites ou moyennes tailles.
- Les mesures géométriques dans l'image ne sont pas très précises.
- Il est difficile d'estimer les grandes surfaces (sol, murs, plafond) avec cette donnée.
