import os
import glob
import numpy as np
import nibabel as nib

def generate_upenn_p2_masks(root_dir):
    """
    Recorre el dataset UPenn-GBM (train y val).
    Toma el Tumor Core del archivo base y lo inyecta en el archivo combined2
    para crear una máscara jerárquica completa para el Pipeline 2.
    """
    for split in ["train", "val"]:
        split_dir = os.path.join(root_dir, split)
        labels_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(labels_dir):
            continue
            
        print(f"\nProcesando partición UPenn: {split.upper()}")
        
        # Buscar todos los archivos base (soportando los dos nombres comunes de UPenn)
        base_files = glob.glob(os.path.join(labels_dir, "*_automated_approx_segm.nii.gz"))
        if not base_files:
            base_files = glob.glob(os.path.join(labels_dir, "*_segm.nii.gz"))
            
        for base_path in base_files:
            # Extraer el ID del caso (ej: UPENN-GBM-00055_11)
            filename = os.path.basename(base_path)
            case_id = filename.replace("_automated_approx_segm.nii.gz", "").replace("_segm.nii.gz", "")
            
            combined_path = os.path.join(labels_dir, f"{case_id}_combined2_approx_segm.nii.gz")
            out_path = os.path.join(labels_dir, f"{case_id}_tumorMask2_approx_segm.nii.gz")
            
            if not os.path.exists(combined_path):
                print(f"  ⚠️ Faltante: No existe combined2 para {case_id}")
                continue
                
            # --- 1. Cargar ambas máscaras ---
            img_base = nib.load(base_path)
            mask_base = img_base.get_fdata().astype(np.uint8)
            
            img_comb = nib.load(combined_path)
            mask_comb = img_comb.get_fdata().astype(np.uint8)
            
            # --- 2. Lógica de Fusión ---
            # Empezamos con una copia de la máscara combinada (que ya tiene Infiltración=6 y Edema=2)
            mask_final = np.copy(mask_comb)
            
            # Extraemos el Tumor Core de la máscara base (Necrosis=1, Enhancing=4)
            tc_mask = (mask_base == 1) | (mask_base == 4)
            
            # Inyectamos el Tumor Core en nuestra máscara final, sobreescribiendo el "agujero"
            mask_final[tc_mask] = mask_base[tc_mask]
            
            # --- 3. Guardar el resultado ---
            nib.save(nib.Nifti1Image(mask_final, img_comb.affine, img_comb.header), out_path)
            print(f"  ✓ {case_id} -> Generada _tumorMask2_approx_segm.nii.gz")

# --- EJECUCIÓN ---
# Ajusta esta ruta a la carpeta raíz de tu dataset de UPenn
UPENN_DATASET_DIR = "./Dataset/Dataset_30_6/"
generate_upenn_p2_masks(UPENN_DATASET_DIR)