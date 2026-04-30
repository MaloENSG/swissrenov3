# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 11:32:32 2026

@author: malo.delacour
"""

import bpy

# =========================================================
# UTILITAIRES
# =========================================================

def clean_scene():
    """Supprime tous les objets de la scène."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_obj(filepath, name):
    """Importe un OBJ et le renomme."""
    bpy.ops.wm.obj_import(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    obj.name = name
    return obj


def apply_scale(obj):
    """Applique les transformations d'échelle."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.select_all(action='DESELECT')


def boolean_operation(target, cutter, operation, name):
    """Applique un booléen (DIFFERENCE ou INTERSECT)."""
    bpy.context.view_layer.objects.active = target
    target.select_set(True)

    mod = target.modifiers.new(name=name, type='BOOLEAN')
    mod.operation = operation
    mod.object = cutter
    mod.solver = 'EXACT'

    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.ops.object.select_all(action='DESELECT')


def clean_mesh(obj, merge_threshold):
    """Nettoyage mesh : merge by distance + delete loose."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=merge_threshold)
    
    # Supprimer géométrie dégénérée (faces ~0 area, edges très courts)
    bpy.ops.mesh.dissolve_degenerate(threshold=merge_threshold)
    
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')


def export_obj(obj, filepath):
    """Exporte un objet en OBJ."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True,
        apply_modifiers=True
    )

    bpy.ops.object.select_all(action='DESELECT')

def separate_loose_parts(obj, base_name):
    """Sépare un objet en plusieurs objets par loose parts."""
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Passer en mode EDIT
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    # récupérer les nouveaux objets créés
    new_objects = [o for o in bpy.context.selected_objects]
    
    # Renommage
    for i, o in enumerate(new_objects):
        o.name = f"{base_name}_{i:03d}"

    bpy.ops.object.select_all(action='DESELECT')
    
    return new_objects

def merge_objects(objects, merged_name):
    """Fusionne une liste d'objets en un seul."""
    bpy.ops.object.select_all(action='DESELECT')

    # Sélectionner les objets
    for obj in objects:
        obj.select_set(True)

    # Définir l'objet actif (le premier de la liste)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()

    merged_obj = bpy.context.view_layer.objects.active
    merged_obj.name = merged_name

    bpy.ops.object.select_all(action='DESELECT')

    return merged_obj

# =========================================================
# PIPELINE PRINCIPAL
# =========================================================


def pipeline_roof(SOLID_PATH, ROOF_PATH, EXPORT_DIFF_PATH, EXPORT_INTER_PATH):
    
    MERGE_THRESHOLD_DIFF  = 0.0015
    MERGE_THRESHOLD_INTER = 0.0025
    SOLID_Z_SCALE_FACTOR  = 3.0
    
    # 1️⃣ Nettoyer la scène
    clean_scene()
    
    # 2️⃣ Importer les objets
    solid = import_obj(SOLID_PATH, "solid_obj")
    roof  = import_obj(ROOF_PATH,  "roof_obj")
    
    if solid.type != 'MESH' or roof.type != 'MESH':
        raise Exception("Les deux objets doivent être des meshes")
    
    # 3️⃣ Ajuster la hauteur du solid (pour assurer une bonne découpe)
    solid.scale.z *= SOLID_Z_SCALE_FACTOR
    bpy.context.view_layer.update()
    apply_scale(solid)
    
    # separation
    roofs = separate_loose_parts(roof, "roof_obj")
    roof_int_list = []
    roof_ext_list = []
    
    for i, ro in enumerate(roofs):
    
        # 4️⃣ Dupliquer roof pour faire INTERSECTION séparément
        roof_inter = ro.copy()
        roof_inter.data = ro.data.copy()
        roof_inter.name = "roof_inter_obj"
        bpy.context.collection.objects.link(roof_inter)
        
        # 5️⃣ BOOLEAN INTERSECTION
        boolean_operation(
            target=roof_inter,
            cutter=solid,
            operation='INTERSECT',
            name="Bool_Intersect"
        )
        
        # 6️⃣ BOOLEAN DIFFERENCE
        boolean_operation(
            target=ro,
            cutter=solid,
            operation='DIFFERENCE',
            name="Bool_Difference"
        )
        
        # 7️⃣ Nettoyage mesh
        clean_mesh(ro, MERGE_THRESHOLD_DIFF)
        clean_mesh(roof_inter, MERGE_THRESHOLD_INTER)
        roof_int_list.append(roof_inter)
        roof_ext_list.append(ro)
        
        
    # fusion
    roof_int = merge_objects(roof_int_list, "roof_int_obj")
    roof_ext = merge_objects(roof_ext_list, "roof_ext_obj")
        
    # 8️⃣ Export des résultats
    export_obj(roof_ext, EXPORT_DIFF_PATH)
    export_obj(roof_int, EXPORT_INTER_PATH)
    
    print("Pipeline terminé avec succès.")



if __name__ == '__main__':

    # =========================================================
    # CONFIGURATION
    # =========================================================
    
    frichename = "FR"
    EGIDname = "190195516"
    
    SOLID_PATH = f"res/{frichename}/bldg_closed_{EGIDname}.obj"
    ROOF_PATH  = f"res/{frichename}/bldg_roof_{EGIDname}.obj"
    
    EXPORT_DIFF_PATH = f"res/{frichename}/bldg_ext_{EGIDname}.obj"
    EXPORT_INTER_PATH = f"res/{frichename}/bldg_int_{EGIDname}.obj"
    
    pipeline_roof(SOLID_PATH, ROOF_PATH, EXPORT_DIFF_PATH, EXPORT_INTER_PATH)
    
    
