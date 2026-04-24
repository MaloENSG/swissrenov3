# 2. Quickstart

## Load LAS file

```python linenums="1"
import numpy as np
import swissrenov3 as s3

pc = s3.IO.read_las("exemple.las")
```

## Analyse PointCloud Structure

```python linenums="1"
# Extraire les données
xyz = pc.xyz
rvb = pc.rvb
print(len(pc))

# Vérifier la classification et l'indexation
pc.is_classified()
pc.is_indexed()

# Ajouter la classification et l'indexation
classif = np.zeros((len(pc),), dtype=int)
pc.classify(classif)
pc.index()
```

## View PointCloudInfo

```python linenums="1"
# Extraire une information
print(pc.info.name)
print(pc.info.source_sensor)
print(pc.info.offset)
...

# Extraire toutes les informations dans un dictionnaire
infos = pc.info.view_data()
print(infos)
```

## Create Referentiel

```python linenums="1"
# Création d'un référentiel
ref = s3.pointcloud.Referentiel(x=2500000.000, 
                   y=1250000.000, 
                   z=400.000, 
                   a=1.32) # en degrées

# Extraire l'offset et l'angle
offset_array = ref.offset()
angle = ref.a
```

## Apply new Referentiel

```python linenums="1"
pc_loc, bbox_glob = s3.geometry.refGlob2refLoc(pc, ref)

print(pc_loc.info.offset) # [2500000, 1250000, 400]
print(pc_loc.info.angle)  # 1.32
```

## Basic selection

```python linenums="1"
idx_in, idx_out = s3.utils.select_crop(pc, x_min = 10, x_max = 25)
pc_cropped = s3.utils.select_pc_index(pc, idx_in)
```

## Write PointCloud in LAS file

```python linenums="1"
pc.write_las("my_pointcloud.las")
```