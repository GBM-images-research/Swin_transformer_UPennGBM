import os
import glob
import numpy as np
import nibabel as nib
import re

def sort_timepoints(tp_list):
    """Ordena los timepoints numéricamente (Timepoint_2 antes que Timepoint_10)."""
    return sorted(tp_list, key=lambda x: int(re.search(r'Timepoint_(\d+)', x).group(1)))

def generate_custom_masks(base_dir):
    """
    Recorre los pacientes en base_dir, genera MÁSCARA 1 para todos los timepoints disponibles,
    y MÁSCARA 2 solo cuando existe un timepoint consecutivo (follow) registrado.
    """
    patient_folders = glob.glob(os.path.join(base_dir, "PatientID_*"))
    
    for pat_path in patient_folders:
        if not os.path.isdir(pat_path):
            continue
            
        patient_id = os.path.basename(pat_path)
        
        # Encontrar y ordenar las carpetas de Timepoints del paciente
        tp_folders = glob.glob(os.path.join(pat_path, "Timepoint_*"))
        tp_folders = sort_timepoints([os.path.basename(p) for p in tp_folders])
        
        if not tp_folders:
            continue
            
        print(f"\nProcesando paciente: {patient_id} ({len(tp_folders)} timepoints)")
        
        # Iterar sobre TODOS los timepoints (ya no nos detenemos en el penúltimo)
        for i in range(len(tp_folders)):
            t_base = tp_folders[i]
            base_mask_file = os.path.join(pat_path, t_base, f"{patient_id}_{t_base}_tumorMask.nii.gz")
            
            if not os.path.exists(base_mask_file):
                print(f"  ⚠️ Faltante: No se encontró la máscara original en {t_base}")
                continue
            
            # --- 1. CARGA DE MÁSCARA BASE ---
            img_base = nib.load(base_mask_file)
            mask_base = img_base.get_fdata().astype(np.uint8)
            affine = img_base.affine
            header = img_base.header
            
            # Definición de etiquetas base (1: Necrosis, 2: Edema, 3: Enhancing Tumor, 4: Cavidad)
            tc_base = (mask_base == 1) | (mask_base == 3)
            edema_base = (mask_base == 2)
            rc_base = (mask_base == 4)
            
            # =================================================================
            # PIPELINE 1: MÁSCARA 1 (Tumor Core vs Edema Total)
            # Se genera SIEMPRE, sin importar si hay un punto temporal siguiente
            # =================================================================
            mask_1 = np.zeros_like(mask_base)
            mask_1[tc_base] = 1      # Clase 1: Tumor Core
            mask_1[edema_base] = 2   # Clase 2: Edema Total
            # La cavidad (rc_base) y el resto quedan en 0 (Background)
            
            out_file_1 = os.path.join(pat_path, t_base, f"{patient_id}_{t_base}_tumorMask_1.nii.gz")
            nib.save(nib.Nifti1Image(mask_1, affine, header), out_file_1)
            print(f"  ✓ {t_base} -> Generada Máscara 1 (P1)")
            
            # =================================================================
            # PIPELINE 2: MÁSCARA 2 (Infiltración vs Edema Puro)
            # Se genera SOLO si existe un T+1 válido y registrado
            # =================================================================
            # Verificamos si no estamos en el último timepoint
            if i < len(tp_folders) - 1:
                t_follow = tp_folders[i+1]
                follow_reg_mask_file = os.path.join(pat_path, t_follow, f"{patient_id}_{t_follow}_tumorMask_flo_reg.nii.gz")
                
                if os.path.exists(follow_reg_mask_file):
                    img_follow = nib.load(follow_reg_mask_file)
                    mask_follow = img_follow.get_fdata().astype(np.uint8)
                    
                    if mask_base.shape == mask_follow.shape:
                        tc_follow = (mask_follow == 1) | (mask_follow == 3)
                        
                        mask_2 = np.zeros_like(mask_base)
                        
                        # Infiltración: TC del follow que NO es TC de la base y NO es cavidad
                        infiltration = tc_follow & (~tc_base) & (~rc_base)
                        # Edema Puro: Edema base que NO se infiltró
                        pure_edema = edema_base & (~infiltration)
                        
                        mask_2[infiltration] = 1 # Clase 1: Infiltración
                        mask_2[pure_edema] = 2   # Clase 2: Edema Puro
                        
                        out_file_2 = os.path.join(pat_path, t_base, f"{patient_id}_{t_base}_tumorMask_2.nii.gz")
                        nib.save(nib.Nifti1Image(mask_2, affine, header), out_file_2)
                        print(f"    ↳ Par válido ({t_base} -> {t_follow}) -> Generada Máscara 2 (P2)")
                    else:
                        print(f"    ❌ Error de shape entre {t_base} y {t_follow}. Omitiendo Máscara 2.")
                else:
                    print(f"    ⚠️ No se encontró máscara registrada de follow en {t_follow}. Omitiendo Máscara 2.")
            else:
                print(f"    ℹ️ Nodo final ({t_base}). No se genera Máscara 2.")

# --- EJECUCIÓN ---
directorios_a_procesar = ['Dataset/MU_glioma/test']

for directorio in directorios_a_procesar:
    if os.path.exists(directorio):
        print(f"\n{'='*50}\nIniciando procesamiento en la partición: {directorio}\n{'='*50}")
        generate_custom_masks(directorio)
    else:
        print(f"\n⚠️ El directorio '{directorio}' no existe en la ruta actual.")